
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
import multiprocessing as mp


# Connection Symbols
def gamma_func(sigma, mu, nu):
    coord_symbols = [t, x, y, z]
    g_sigma_mu_nu = 0
    for rho in [0,1,2,3]:
        if g[sigma, rho] != 0:
            g_sigma_mu_nu += 1/2 * 1/g[sigma, rho] * (\
                            g[nu, rho].diff(coord_symbols[mu]) + \
                            g[rho, mu].diff(coord_symbols[nu]) - \
                            g[mu, nu].diff(coord_symbols[rho]) )
        else:
            g_sigma_mu_nu += 0
    return g_sigma_mu_nu

metric = "schwarzschild"
M = 1.0
r_s_value = 2*M 

# Type of geodesic
time_like = False # No Massive particle geodesics yet

# Define symbolic variables
t, x, y, z, r_s = sp.symbols('t x y z r_s')
        
# Radial distance to BlackHole location
R = sp.sqrt(x**2 + y**2 + z**2)

# The implemented metrics:
if metric == "flat":
    g = sp.Matrix([\
        [-1, 0, 0, 0],\
        [0, 1, 0, 0],\
        [0, 0, 1, 0],\
        [0, 0, 0, 1]\
        ])
elif metric == "schwarzschild":
    g = sp.Matrix([\
        [-(1-r_s/(4*R))**2 / (1+r_s/(4*R))**2, 0, 0, 0],\
        [0, (1+r_s/(4*R))**4, 0, 0], \
        [0, 0, (1+r_s/(4*R))**4, 0], \
        [0, 0, 0, (1+r_s/(4*R))**4], \
      ])     
        
# Connection Symbols
gam_t = sp.Matrix([[gamma_func(0,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
gam_x = sp.Matrix([[gamma_func(1,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
gam_y = sp.Matrix([[gamma_func(2,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
gam_z = sp.Matrix([[gamma_func(3,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])

# Building up the geodesic equation: 
# Derivatives: k_beta = d x^beta / d lambda
k_t, k_x, k_y, k_z = sp.symbols('k_t k_x k_y k_z', real=True)
k = [k_t, k_x, k_y, k_z]

# Second derivatives: d k_beta = d^2 x^beta / d lambda^2
dk_t = sum([- gam_t[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
dk_x = sum([- gam_x[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
dk_y = sum([- gam_y[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
dk_z = sum([- gam_z[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])

# Norm of k
# the norm of k determines if you have a massive particle (-1), a mass-less photon (0) 
# or a space-like curve (1)
norm_k = g[0, 0]*k_t**2 + g[1,1]*k_x**2 + \
                g[2,2]*k_y**2 + g[3,3]*k_z**2

# Now we calculate k_t using the norm. This eliminates one of the differential equations.
# time_like = True: calculates a geodesic for a massive particle (not implemented yet)
# time_like = False: calculates a geodesic for a photon
if (time_like):
    k_t_from_norm = sp.solve(norm_k+1, k_t)[1]
else:
    k_t_from_norm = sp.solve(norm_k, k_t)[1]

# Lambdify versions
dk_x_lamb = sp.lambdify([k_x, x, k_y, y, k_z, z, \
                              k_t, t, r_s], \
                             dk_x, "numpy")
dk_y_lamb = sp.lambdify([k_x, x, k_y, y, k_z, z, \
                              k_t, t, r_s], \
                             dk_y, "numpy")
dk_z_lamb = sp.lambdify([k_x, x, k_y, y, k_z, z, \
                              k_t, t, r_s], \
                             dk_z, "numpy")
k_t_from_norm_lamb = sp.lambdify([k_x, x, k_y, y, k_z, z, \
                                       r_s], k_t_from_norm, "numpy")
 


# This function does the numerical integration of the geodesic equation using scipy's solve_ivp
def calc_trajectory(k_x_0 = 1., x0 = -10.0, k_y_0 = 0., y0 = 5.0, k_z_0 = 0., z0 = 5.0,\
                    R_end = -1,\
                    curve_start = 0, \
                    curve_end = 50, \
                    nr_points_curve = 50, \
                    max_step = np.inf,\
                    verbose = False \
                   ):

    # IMPLEMENT: check if inside horizon!
    r0 = np.linalg.norm(np.array([x0, y0, z0]))
    if r0 > r_s_value:
        # Step function needed for solve_ivp
        def step(lamb, new):
            new_k_x, new_x, new_k_y, new_y, new_k_z, new_z = new

            new_k_t = k_t_from_norm_lamb(*new, r_s_value)
            new_dk_x = dk_x_lamb(*new, new_k_t, t = 0, r_s = r_s_value)
            dx = new_k_x
            new_dk_y = dk_y_lamb(*new, new_k_t, t = 0, r_s = r_s_value)
            dy = new_k_y
            new_dk_z = dk_z_lamb(*new, new_k_t, t = 0, r_s = r_s_value)
            dz = new_k_z

            return( new_dk_x, dx, new_dk_y, dy, new_dk_z, dz)

        def hit_blackhole(t, y): 
            k_x, x, k_y, y, k_z, z = y
            if verbose: print("Test Event Hit BH: ", x, y, z, r_s_value, x**2 + y**2 + z**2 - r_s_value**2)
            return x**2 + y**2 + z**2 - r_s_value**2
        hit_blackhole.terminal = True

        def reached_end(t, y): 
            k_x, x, k_y, y, k_z, z = y
            if verbose: print("Test Event End: ", np.sqrt(x**2 + y**2 + z**2), R_end, x**2 + y**2 + z**2 - R_end**2)
            return x**2 + y**2 + z**2 - R_end**2
        reached_end.terminal = True
        
        values_0 = [ k_x_0, x0, k_y_0, y0, k_z_0, z0 ]
        if nr_points_curve == 0:
            t_pts = None
        else:
            t_pts = np.linspace(curve_start, curve_end, nr_points_curve)

        start = time.time()
        events = [hit_blackhole]
        if R_end > r0 : events.append(reached_end)
        result = solve_ivp(step, (curve_start, curve_end), values_0, t_eval=t_pts, \
                           events=events,\
                           max_step = max_step)
        end = time.time()
        if verbose: print("New: ", result.message, end-start, "sec")


        result.update({"hit_blackhole": len(result.t_events[0])>0})
        result.update({"start_inside_hole": False})

    else:
        if verbose: print("Starting location inside the blackhole.")
        result = {"start_inside_hole": True}

    return result


def wrap_calc_trajectory(x0, y0=2):
    #print("  running wrap")
    return calc_trajectory(k_x_0 = 1., x0 = x0, k_y_0 = 0., y0 = 5.0, k_z_0 = 0., z0 = 5.0,\
                    R_end = -1,\
                    curve_start = 0, \
                    curve_end = 50, \
                    nr_points_curve = 50, \
                    max_step = np.inf,\
                    verbose = False \
                   )



if __name__ == '__main__':
    for nr in [2*512*512]:
        todo = [-20 for i in range(nr)]

        def test1():
            for x0 in todo:#, -20]:
                wrap_calc_trajectory(x0, y0=2)

        _start = time.time()
        test1()
        test1_time = time.time()-_start
        #print("Test1: ", test1_time)

        _start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = [pool.apply(wrap_calc_trajectory, args=(x0,)) for x0 in todo]#, -20]]
        print(len(results))
        pool.close()    

        #print(results)
        test_par_time = time.time()-_start
        #print("TIME 2: ", test_par_time)
        print("SPEEDUP: ", test1_time/test_par_time, "for: ", nr, "calcs")



    # # This function does the numerical integration of the geodesic equation using scipy's solve_ivp
    # def calc_trajectory_mp(self, \
    #                     k_x_0 = [1.], x0 =[-10.0], k_y_0 = [0.], y0 = [5.0], k_z_0 = [0.], z0 = [5.0],\
    #                     R_end = -1,\
    #                     curve_start = 0, \
    #                     curve_end = 50, \
    #                     nr_points_curve = 50, \
    #                     max_step = np.inf,\
    #                     verbose = False \
    #                    ):

    #     # IMPLEMENT: check if inside horizon!
    #     r0 = np.linalg.norm(np.array([x0, y0, z0]))
    #     if r0 > r_s_value:
    #         # Step function needed for solve_ivp
    #         global step
    #         def step(lamb, new):
    #             new_k_x, new_x, new_k_y, new_y, new_k_z, new_z = new

    #             new_k_t = k_t_from_norm_lamb(*new, r_s_value)
    #             new_dk_x = dk_x_lamb(*new, new_k_t, t = 0, r_s = r_s_value)
    #             dx = new_k_x
    #             new_dk_y = dk_y_lamb(*new, new_k_t, t = 0, r_s = r_s_value)
    #             dy = new_k_y
    #             new_dk_z = dk_z_lamb(*new, new_k_t, t = 0, r_s = r_s_value)
    #             dz = new_k_z

    #             return( new_dk_x, dx, new_dk_y, dy, new_dk_z, dz)

    #         global hit_blackhole
    #         def hit_blackhole(t, y): 
    #             k_x, x, k_y, y, k_z, z = y
    #             if verbose: print("Test Event Hit BH: ", x, y, z, r_s_value, x**2 + y**2 + z**2 - r_s_value**2)
    #             return x**2 + y**2 + z**2 - r_s_value**2
    #         hit_blackhole.terminal = True

    #         global reached_end
    #         def reached_end(t, y): 
    #             k_x, x, k_y, y, k_z, z = y
    #             if verbose: print("Test Event End: ", np.sqrt(x**2 + y**2 + z**2), R_end, x**2 + y**2 + z**2 - R_end**2)
    #             return x**2 + y**2 + z**2 - R_end**2
    #         reached_end.terminal = True
            
    #         values_0 = zip(*[ k_x_0, x0, k_y_0, y0, k_z_0, z0 ])
    #         if nr_points_curve == 0:
    #             t_pts = None
    #         else:
    #             t_pts = np.linspace(curve_start, curve_end, nr_points_curve)

    #         start = time.time()
    #         events = [hit_blackhole]
    #         if R_end > r0 : events.append(reached_end)

    #         arguments = []
    #         for v0 in values_0:
    #             arguments.append((step, (curve_start, curve_end), v0, t_pts, events, max_step))

    #         # Argument wrapper
    #         global _solve_ivp
    #         def _solve_ivp(step, t_span, v0, t_pts, events, max_step):
    #             return solve_ivp(step, t_span, v0, t_eval=t_pts, \
    #                            events=events,\
    #                            max_step = max_step)
    #         print("STARTING")
    #         with mp.Pool(processes=1) as pool:
    #             results = pool.starmap(_solve_ivp, tuple(arguments))
    #         print("DONE:")
    #         # result = solve_ivp(step, (curve_start, curve_end), values_0, t_eval=t_pts, \
    #         #                    events=events,\
    #         #                    max_step = max_step)
    #         end = time.time()
    #         if verbose: print("New: ", result.message, end-start, "sec")


    #         result.update({"hit_blackhole": len(result.t_events[0])>0})
    #         result.update({"start_inside_hole": False})

    #     else:
    #         if verbose: print("Starting location inside the blackhole.")
    #         result = {"start_inside_hole": True}

    #     return result


    # def calc_trajectory_2(self, \
    #                     s0 = [1., -10.0, 0, 5.0, 0, 5.0],\
    #                     R_end = -1,\
    #                     curve_start = 0, \
    #                     curve_end = 50, \
    #                     nr_points_curve = 50, \
    #                     max_step = np.inf,\
    #                     verbose = False \
    #                    ):

    #     k_x_0, x0, k_y_0, y0, k_z_0, z0 = s0
    #     return calc_trajectory(k_x_0, x0, k_y_0, y0, k_z_0, z0,\
    #                     R_end, curve_start, curve_end, nr_points_curve, max_step, verbose)

    # # This function does the numerical integration of the geodesic equation using scipy's solve_ivp
    # def calc_trajectories(self, \
    #                     k_x_0 = [1.], x0 = [-10.0], k_y_0 = [0.], y0 = [5.0], k_z_0 = [0.], z0 = [5.0],\
    #                     R_end = -1,\
    #                     curve_start = 0, \
    #                     curve_end = 50, \
    #                     nr_points_curve = 50, \
    #                     max_step = np.inf,\
    #                     verbose = False \
    #                    ):

    #     start0_list = list(zip(*[k_x_0, x0, k_y_0, y0, k_z_0, z0]))
    #     pool = mp.Pool(mp.cpu_count())
    #     print([start0 for start0 in start0_list])
    #     results = [pool.apply(calc_trajectory_2, args=(self, start0)) for start0 in start0_list]

    #     # Step 3: Don't forget to close
    #     pool.close()   

    #     print(results) 


