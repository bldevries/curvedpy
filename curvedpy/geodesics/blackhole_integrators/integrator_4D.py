import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time


class Integrator4D:

    def __init__(self):
        pass

    ################################################################################################
    # 
    ################################################################################################
    # This function does the numerical integration of the geodesic equation using scipy's solve_ivp
    def integrate(self, \
                        k4_start, x4_start, \
                        get_dk,\
                        hit_blackhole,\
                        stop_integration = None,\
                        stop_integration_coord_check = None,\
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
        
        # Step function needed for solve_ivp
        def step(lamb, new):
            k_0_new, k_1_new, k_2_new, k_3_new, x_0_new, x_1_new, x_2_new, x_3_new = new

            dk_0, dk_1, dk_2, dk_3 = get_dk(k_0_new, k_1_new, k_2_new, k_3_new, x_0_new, x_1_new, x_2_new, x_3_new)
            dx_0, dx_1, dx_2, dx_3 = k_0_new, k_1_new, k_2_new, k_3_new

            return( dk_0, dk_1, dk_2, dk_3, dx_0, dx_1, dx_2, dx_3)

        # EVENTS
        # This is not perfectly general yet!!
        events = []

        hit_blackhole.terminal = True
        events.append(hit_blackhole)

        if stop_integration:
            stop_integration.terminal = True
            events.append(stop_integration)

        if stop_integration_coord_check:
            stop_integration_coord_check.terminal = True
            events.append(stop_integration_coord_check)

        values_0 = [ *k4_start, *x4_start ]

        if nr_points_curve == 0:
            t_pts = None
        else:
            t_pts = np.linspace(curve_start, curve_end, nr_points_curve)

        start = time.time()
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
        if stop_integration:
            result.update({"end_check": len(result.t_events[1])>0})
        if stop_integration_coord_check:
            result.update({"stop_integration_coord_check": len(result.t_events[2])>0})

        return result