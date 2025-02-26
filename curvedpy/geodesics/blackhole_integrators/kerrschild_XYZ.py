import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
from curvedpy.metrics.kerrschild_metric import KerrSchildMetricXYZ
#from curvedpy.metrics.schwarzschild_metric_AUTOGRAD import SchwarzschildMetricXYZ_AUTOGRAD

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
class GeodesicIntegratorKerrSchildXYZ:

    conversions = Conversions()

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, a = 0.1, time_like = False, verbose=False):

        self.M = mass
        self.a = a

        # Type of geodesic
        self.time_like = time_like # No Massive particle geodesics yet
        self.verbose = verbose
        if verbose:
            print("Geodesic SS Integrator Settings: ")
            print(f"  - {self.M=}")
            print(f"  - {self.a=}")
            print(f"  - {self.time_like=}")
            print(f"  - {self.verbose=}")
            print("--")

        self.metric = KerrSchildMetricXYZ(mass, a, time_like, verbose)

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
                        verbose = False, *args, **kargs \
                       ):

        # Calculate from norm of starting condition
        k0_t = self.metric.k_t_from_norm(k0=k0_xyz, x0= x0_xyz, t = 0)
        values_0 = [k0_t, *k0_xyz, *x0_xyz]


        r0 = np.linalg.norm(x0_xyz)
        if R_end == -1:
            R_end = np.inf
        elif R_end < r0:
            R_end = np.sqrt(x0**2 + y0**2 + z0**2)*1.01

        if self.metric.check_horizin(*x0_xyz) > 0.0:
            # Step function needed for solve_ivp
            def step(lamb, new):
                k_t_new, k_x_new, k_y_new, k_z_new, x_new, y_new, z_new = new

                dk_t, dk_x, dk_y, dk_z = self.metric.get_dk(k_t_new, k_x_new, k_y_new, k_z_new, \
                                                    t_val=lamb, x_val=x_new, y_val=y_new, z_val=z_new)

                dx, dy, dz = k_x_new, k_y_new, k_z_new
                return( dk_t, dk_x, dk_y, dk_z, dx, dy, dz)

            def hit_blackhole(t, y): 
                eps = 0.01
                k_t, k_x, k_y, k_z, x, y, z = y
                return self.metric.check_horizin(x, y, z)
                #return np.sqrt(x**2+y**2+z**2) - (self.r_s_value+eps)
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

