import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
from curvedpy import Conversions

class GeodesicIntegratorSchwarzschild:

    conversions = Conversions()


    def __init__(self, mass=1.0, verbose=False):
        self.M = mass
        self.r_s_value = 2*self.M 

        # Type of geodesic
        self.time_like = False # No Massive particle geodesics yet

        # Define symbolic variables
        self.t, self.r, self.th, self.ph, self.r_s = sp.symbols("t r \\theta \\phi r_s")

        self.g = sp.Matrix([\
            [-1*(1-self.r_s/self.r), 0, 0, 0],\
            [0, 1/(1-self.r_s/self.r), 0, 0],\
            [0, 0, self.r**2, 0],\
            [0, 0, 0, self.r**2 * sp.sin(self.th)**2]\
            ])

        # Connection Symbols
        self.gam_t = sp.Matrix([[self.gamma_func(0,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_r = sp.Matrix([[self.gamma_func(1,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_th = sp.Matrix([[self.gamma_func(2,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_ph = sp.Matrix([[self.gamma_func(3,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
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







    # Connection Symbols
    def gamma_func(self, sigma, mu, nu):
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


    ################################################################################################
    #
    ################################################################################################
    def calc_trajectory(self, \
                        k0_xyz = [1, 0.0, 0.0], x0_xyz = [-10, 10, 0], \
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

        x0_sph, k0_sph = self.conversions.convert_xyz_to_sph(x0_xyz, k0_xyz)
        k_r_0, k_th_0, k_ph_0 = k0_sph
        r0, th0, ph0 = x0_sph

        result = self.calc_trajectory_sph(\
                        k_r_0 = k_r_0, r0 = r0, k_th_0=k_th_0, th0=th0, k_ph_0=k_ph_0, ph0=ph0,\
                        R_end = R_end, curve_start = curve_start, curve_end = curve_end, nr_points_curve = nr_points_curve, \
                        method = method, max_step = max_step, first_step = first_step, rtol = rtol, atol = atol,\
                        verbose = verbose )
                       

        k_r, r, k_th, th, k_ph, ph = result.y
        k_sph = list(zip(*[k_r, k_th, k_ph]))
        x_sph = list(zip(*[r, th, ph]))

        # THIS IS REDICULOUSLY SLOW, FIX IT!
        k_xyz, x_xyz = [], []
        for i in range(len(k_sph)):
            x, k = self.conversions.convert_sph_to_xyz(x_sph[i], k_sph[i])

            # k, x = self.convert_sph_to_xyz(k_sph[i], x_sph[i])
            k_xyz.append(k)
            x_xyz.append(x)

        return list(zip(*k_xyz)), list(zip(*x_xyz)), result #k_xyz, x_xyz , result


    ################################################################################################
    #
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

        if r0 > self.r_s_value:
            # Step function needed for solve_ivp
            def step(lamb, new):

                new_k_r, new_r, new_k_th, new_th, new_k_ph, new_ph = new
                new_k_t = self.k_t_from_norm_lamb(*new, self.r_s_value)

                #print(np.sqrt(new_x**2 + new_y**2 + new_z**2))

                new_dk_r = self.dk_r_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dr = new_k_r
                new_dk_th = self.dk_th_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dth = new_k_th
                new_dk_ph = self.dk_ph_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dph = new_k_ph
                #print("step", new_x, new_y, new_z, dx, dy, dz)

                return( new_dk_r, dr, new_dk_th, dth, new_dk_ph, dph)

            def hit_blackhole(t, y): 
                eps = 0.5
                k_r, r, k_th, th, k_ph, ph = y
                #if verbose: print("Test Event Hit BH: ", x, y, z, self.r_s_value, x**2 + y**2 + z**2 - self.r_s_value**2)
                return r - self.r_s_value
            hit_blackhole.terminal = True

            # def reached_end(t, y): 
            #     k_x, x, k_y, y, k_z, z = y
            #     if verbose: print("Test Event End: ", np.sqrt(x**2 + y**2 + z**2), R_end, x**2 + y**2 + z**2 - R_end**2)
            #     return x**2 + y**2 + z**2 - R_end**2
            # reached_end.terminal = True
            
            values_0 = [ k_r_0, r0, k_th_0, th0, k_ph_0, ph0 ]
            if nr_points_curve == 0:
                t_pts = None
            else:
                t_pts = np.linspace(curve_start, curve_end, nr_points_curve)

            start = time.time()
            events = [hit_blackhole]
            if R_end > r0 : events.append(reached_end)
            result = solve_ivp(step, (curve_start, curve_end), values_0, t_eval=t_pts, \
                               events=events,\
                               method=method,\
                               max_step = max_step,\
                               first_step = first_step,\
                               atol=atol,\
                               rtol = rtol)
            end = time.time()
            if verbose: print("New: ", result.message, end-start, "sec")


            result.update({"hit_blackhole": len(result.t_events[0])>0})
            result.update({"start_inside_hole": False})

        else:
            if verbose: print("Starting location inside the blackhole.")
            result = {"start_inside_hole": True}

        return result


    # def convert_to_xyz(self, r, th, ph):
    #     z = r*np.cos(th)
    #     x = r*np.sin(th)*np.cos(ph)
    #     y = r*np.sin(th)*np.sin(ph)
    #     return x, y, z

    # def convert_to_sph(self, x, y, z):
    #     r = np.sqrt(x**2 + y**2 + z**2)
    #     th = np.acos(z/r)
    #     ph = np.atan2(y, x) #ph = np.atan(y/x)
    #     return r, th, ph


    # def convert_xyz_to_sph(self, k_xyz, x_xyz ):
    #     k_x, k_y, k_z = k_xyz
    #     x_val, y_val, z_val = x_xyz
    #     #r_val, th_val, ph_val
    #     x_sph = self.convert_to_sph(x_val, y_val, z_val)

    #     x, y, z = sp.symbols(" x y z ")

    #     r = sp.sqrt(x**2+y**2+z**2)
    #     th = sp.acos(z/r)
    #     phi = sp.atan(y/x)

    #     M_xyz_to_sph = sp.Matrix([[r.diff(x), r.diff(y), r.diff(z)],\
    #                               [th.diff(x), th.diff(y), th.diff(z)],\
    #                               [phi.diff(x), phi.diff(y), phi.diff(z)],\
    #                              ])

    #     k = sp.Matrix([k_x, k_y, k_z])

    #     k_sph = M_xyz_to_sph*k
    #     k_sph = k_sph.subs(x, x_val).subs(y, y_val).subs(z, z_val)
    #     #k_r, k_th, k_ph = list(k_sph)

    #     return list(k_sph), x_sph
    #     k_r, r_val, k_th, th_val, k_ph, ph_val

    # def convert_sph_to_xyz(self, k_sph, x_sph):
    #     k_r, k_th, k_ph = k_sph
    #     r_val, th_val, ph_val = x_sph
    #     #x_val, y_val, z_val
    #     x_xyz = self.convert_to_xyz(r_val, th_val, ph_val)

    #     #r_val, th_val, ph_val = self.convert_to_sph(x_val, y_val, z_val)

    #     r, th, ph = sp.symbols(" r \\theta \\phi ")

    #     x = r * sp.sin(th) * sp.cos(ph)
    #     y = r * sp.sin(th) * sp.sin(ph)
    #     z = r * sp.cos(th)

    #     M_sph_to_xyz = sp.Matrix([[x.diff(r), x.diff(th), x.diff(ph)],\
    #                               [y.diff(r), y.diff(th), y.diff(ph)],\
    #                               [z.diff(r), z.diff(th), z.diff(ph)],\
    #                              ])

    #     k = sp.Matrix([k_r, k_th, k_ph])

    #     k_xyz = M_sph_to_xyz*k
    #     k_xyz = k_xyz.subs(r, r_val).subs(th, th_val).subs(ph, ph_val)
    #     k_x, k_y, k_z = list(k_xyz)

    #     return list(k_xyz), x_xyz #k_x, x_val, k_y, y_val, k_z, z_val




