import sympy as sp
import numpy as np
#from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import time
from curvedpy.utils.conversions import Conversions
from curvedpy.geodesics.blackhole_integrators.integrator_4D import Integrator4D
from curvedpy.metrics.schwarzschild_metric import SchwarzschildMetricSpherical

from curvedpy.utils.conversions_SPH2PATCH_3D import cart_to_sph2, cart_to_sph1, sph1_to_cart, sph2_to_cart, sph1_to_sph2, sph2_to_sph1, coords_sph1_to_cart, coords_sph2_to_cart

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
class GeodesicIntegratorSchwarzschildSPH1SPH2:

    conversions = Conversions()
    integrator = Integrator4D()

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, time_like = False, theta_switch = 0.1*np.pi, eps_theta=0.0000001, verbose=False):
        self.M = mass
        self.theta_switch = theta_switch
        self.metric = SchwarzschildMetricSpherical(mass=mass, eps_theta=eps_theta)
        self.verbose = verbose

    ################################################################################################
    #
    ################################################################################################
    # This function checks the shape of the parameters and if multiple photons need to be run
    def calc_trajectory(self, k0_xyz, x0_xyz, full_save = False, *args, **kargs):

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
            return self.calc_trajectory_2patch(k0_xyz[0], x0_xyz[0], full_save, *args, **kargs)
        else:
            return [self.calc_trajectory_2patch(k0_xyz[i], x0_xyz[i], full_save, *args, **kargs) for i in range(len(x0_xyz))]

    
    ################################################################################################
    # 
    ################################################################################################
    # This function handles the switching between the different coordinate patches
    def calc_trajectory_2patch(self, k0_xyz, x0_xyz, full_save = False, *args, **kargs):
        if not isinstance(x0_xyz,np.ndarray):
            x0_xyz = np.array(x0_xyz)

        if not isinstance(k0_xyz,np.ndarray):
            k0_xyz = np.array(k0_xyz)

        summary = {"coords":[]}

        # We start by assuming spherical coordinate patch 1
        # when theta is measured from the z-axis
        k0_sph, x0_sph = cart_to_sph1(k0_xyz, x0_xyz)
        coord = "sph1"
        if x0_sph[1] <= self.theta_switch:
            # Switch to second patch 2 if close to the theta-axis
            k0_sph, x0_sph = cart_to_sph2(k0_xyz, x0_xyz)
            coord = "sph2"
        summary["coords"].append(coord)

        # The geodesic pieces calculated in SPH1 or SPH2 will be stored in this list and
        # in cartesian coordinates.
        final_k_xyz, final_x_xyz = [], []
        # Here the propertime/parameter of the geodesic is tracked
        final_t = []
        t_end_prev_piece = 0.0
        final_k_t = []
        # The integrator information for every piece is tracked in this list
        results_list = []

        # Run the first integration for the first geodesic piece 
        if self.verbose: print(f"=> First run in {coord=}")
        k_sph_step, x_sph_step, res = self.calc_trajectory_sph(k0_sph, x0_sph, coord, *args, **kargs)
        k_t, k_r, k_th, k_ph, t, r, th, ph = res.y

        # Convert the first piece to cartesian and store it
        if coord == "sph1":
            k_xyz_step, x_xyz_step = sph1_to_cart(k_sph_step, x_sph_step)
        else: # coords == "sph2":
            k_xyz_step, x_xyz_step = sph2_to_cart(k_sph_step, x_sph_step)

        # NEED TO TRACK res.t! 
        final_t.append(np.array(t))#[1:]))#+t_end_prev_piece) # We do not need the first entry
        final_k_t.append(k_t)#[1:])

        # Save the used coordinate
        res.coord = coord
        # Save the geodesic piece
        res.x_xyz = x_xyz_step
        res.k_xyz = k_xyz_step
        # Add geodesic and result dict to final results
        results_list.append(res)
        final_k_xyz.append(np.column_stack(k_xyz_step))#[1:])
        final_x_xyz.append(np.column_stack(x_xyz_step))#[1:])

        # Now we will check if we need to integrate another piece or 
        # if we are done.
        nr_switches = 0
        while res.stop_integration_coord_check and (not res.hit_blackhole) and nr_switches <=4:
            # We start the next piece where the previous ended
            k_sph, x_sph = np.column_stack(k_sph_step), np.column_stack(x_sph_step)
            k0_sph, x0_sph = k_sph[-1], x_sph[-1]

            # We convert the end point of the previous piece to the 
            # other coordinate patch
            if coord == "sph1":
                k0_sph, x0_sph = sph1_to_sph2(k0_sph, x0_sph)
                coord = "sph2"
            else: # coord == "sph2"
                k0_sph, x0_sph = sph2_to_sph1(k0_sph, x0_sph)
                coord = "sph1"
            summary["coords"].append(coord)
            if self.verbose: print(f"=> Starting {nr_switches} ({res.stop_integration_coord_check}, {res.hit_blackhole})")
            if self.verbose: print(f" th0 = {round(x0_sph[1],2)} in {coord}")

            # Then we run the next piece
            k_sph_step, x_sph_step, res = self.calc_trajectory_sph(k0_sph, x0_sph, coord, use_t0_k_t_0 = True, t0 = t[-1], k_t_0 = k_t[-1], *args, **kargs)
            k_t, k_r, k_th, k_ph, t, r, th, ph = res.y

            # We convert the new piece to cartesian
            if coord == "sph1":
                k_xyz_step, x_xyz_step = sph1_to_cart(k_sph_step, x_sph_step)
            else: # coords == "sph2":
                k_xyz_step, x_xyz_step = sph2_to_cart(k_sph_step, x_sph_step)
            
            # Track the propertime/parameter and k_t
            final_t.append(t[1:])#+t_end_prev_piece) # We do not need the first entry
            final_k_t.append(k_t[1:])

            # We save all the results
            res.coord = coord
            res.x_xyz = x_xyz_step
            res.k_xyz = k_xyz_step
            results_list.append(res)
            final_k_xyz.append(np.column_stack(k_xyz_step)[1:])
            final_x_xyz.append(np.column_stack(x_xyz_step)[1:])

            # And increase the counter
            nr_switches += 1


        # Merge all pieces
        merged_k_xyz = np.column_stack(np.concatenate(final_k_xyz))
        merged_x_xyz = np.column_stack(np.concatenate(final_x_xyz))
        # Merge the propertime/parameter of the different pieces
        merged_t = np.concatenate(final_t)
        merged_k_t = np.concatenate(final_k_t)

        summary["t"] = merged_t
        summary["k_t"] = merged_k_t
        # If the last piece hits the blackhole the geodesic ends there
        summary["hit_blackhole"] = res["hit_blackhole"] 

        if full_save: 
            print(" == Carefull, doing full saving in schwarzschild_SPH1SPH2.py. This results in huge files for debugging == ")
            summary["result_list"] = results_list

        return merged_k_xyz, merged_x_xyz, summary


    ################################################################################################
    #
    ################################################################################################
    # This function calculates one piece of the geodesic in one coordinate patch
    def calc_trajectory_sph(self, \
                        k0_sph, x0_sph, \
                        coord,\
                        use_t0_k_t_0 = False, t0=0.0, k_t_0=0.0,\
                        R_end = 100,\
                        curve_start = 0, \
                        curve_end = 10000, \
                        nr_points_curve = 5000, \
                        method = "RK45",\
                        max_step = np.inf,\
                        first_step = None,\
                        rtol = 1e-3,\
                        atol = 1e-6,\
                        verbose = False \
                       ):

        k_r_0, k_th_0, k_ph_0 = k0_sph
        r0, th0, ph0 = x0_sph

        # Calculate from norm of starting condition
        if not use_t0_k_t_0:
            t0 = 0
            k_t_0 = self.metric.k_t_from_norm_lamb(k_r_0, k_th_0, k_ph_0, t0, r0, th0, ph0)

        k_0 = np.array([k_t_0, *k0_sph])
        x_0 = np.array([t0, *x0_sph])

        #Check if starting values are outside the blackhole
        if r0 < self.metric.get_r_s():
            print("Starting value inside blackhole")
            return

        def hit_blackhole(t, y): 
            eps = 0.01
            k_0, k_1, k_2, k_3, x_0, x_1, x_2, x_3 = y
            r = x_1#calc_radius_from_x_mu(x_0, x_1, x_2, x_3)
            return r - (self.metric.get_r_s()+eps)

        if R_end == -1: R_end = np.inf
        #elif R_end < r0: R_end = r0*1.01
        def stop_integration(t, y): 
            k_0, k_1, k_2, k_3, x_0, x_1, x_2, x_3 = y
            r = x_1 #calc_radius_from_x_mu(x_0, x_1, x_2, x_3)
            return r - R_end

        def switch_coordinates(t, y):
            if self.theta_switch < 0.0:
                return 1
            else:
                k_0, k_1, k_2, k_3, x_0, x_1, x_2, x_3 = y
                if coord == "sph1":
                    x, y, z = coords_sph1_to_cart(x_1, x_2, x_3)
                    th = np.acos(z/np.sqrt(x**2+y**2+z**2))
                    #print("sph1", th/np.pi)
                    # if abs(th) <= self.theta_switch:# and abs(th) >= 2*np.pi-self.theta_switch:
                    #     return 0
                    # else:
                    #     return 1
                else:
                    x, y, z = coords_sph2_to_cart(x_1, x_2, x_3)
                    th = np.acos(y/np.sqrt(x**2+y**2+z**2))
                    #print(coord, "=?sph2", th/np.pi)
                    # if abs(th) >= np.pi-self.theta_switch:# and abs(th) <= np.pi+self.theta_switch:
                    #     return 0
                    # else:
                    #     return 1
                
                if abs(th) <= self.theta_switch:# and abs(th) >= 2*np.pi-self.theta_switch:
                    return 0
                elif abs(th) >= np.pi-self.theta_switch:# and abs(th) <= np.pi+self.theta_switch:
                    return 0
                else:
                    return 1

                #return th - self.theta_switch

        result = self.integrator.integrate(\
                        k_0, x_0, self.metric.get_dk, hit_blackhole, \
                        stop_integration = stop_integration,\
                        stop_integration_coord_check = switch_coordinates,\
                        curve_start = curve_start, curve_end = curve_end, nr_points_curve = nr_points_curve, \
                        method = method, \
                        max_step = max_step, first_step = first_step, \
                        rtol = rtol, atol = atol,\
                        verbose = verbose )
                       

        k_t, k_r, k_th, k_ph, t, r, th, ph = result.y
        lamb = result.t

        k_sph = np.array([k_r, k_th, k_ph])
        x_sph = np.array([r, th, ph])

        k4_sph = np.array([k_t, k_r, k_th, k_ph])
        x4_sph = np.array([t, r, th, ph])

        result.update({"k4_sph": k4_sph, "x4_sph": x4_sph})

        return k_sph, x_sph, result

