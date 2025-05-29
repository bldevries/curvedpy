import sympy as sp
import numpy as np
#from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import time
from curvedpy.utils.conversions import Conversions
from curvedpy.geodesics.blackhole_integrators.integrator_4D import Integrator4D
from curvedpy.metrics.schwarzschild_metric import SchwarzschildMetricSpherical

from curvedpy.utils.conversions_4D import Conversions4D

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
class IntegratorSchwarzschildSPH2D:

    conversions = Conversions()
    integrator = Integrator4D()


    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, time_like = False, eps_theta=0.0000001, verbose=False):
        self.metric = SchwarzschildMetricSpherical2D(mass=mass)
        self.lpnCoords = LPN_coordinates()

    # def geodesic(self, k0_xyz, x0_xyz):
    #     if not isinstance(k0_xyz, np.ndarray): k0_xyz = np.array(k0_xyz)
    #     if not isinstance(x0_xyz, np.ndarray): x0_xyz = np.array(x0_xyz)
    #     if k0_xyz.shape != x0_xyz.shape:
    #         print("k and x are not the same shape")
    #         return
    #     if k0_xyz.ndim == 1:
    #         if k0_xyz.shape[0] != 3 or x0_xyz.shape[0] != 3:
    #             print("k or x do not have 3 components")
    #             return
    #         k0_xyz = k0_xyz.reshape(1,3)
    #         x0_xyz = x0_xyz.reshape(1,3)

    #     else:
    #         if k0_xyz.shape[1] != 3 or x0_xyz.shape[1] != 3:
    #             print("k or x do not have 3 components")
    #             return



        #k0_lp = np.array([1, 0]).T

        # k0_sph2, x0_sph2 = self.lpnCoords.lp_to_sph2(x0_lp)

        # k_lp, x_lp = self.lpnCoords.sph2_to_lp(k0_sph2, x0_sph2)

        # If p==0 we have a singular transformation matrix, since the 
        # basis vector for p is arbitrary and not restricted. This is when the
        # ray is directed precisely at the center of the BH
        # l==0 is no problem, since we take the direction k of the ray 
        # to define the unit vector in the l direction
        # if np.any((x0_lp.T)[1] == 0.0): print("SINGULAR", (x0_lp.T)[1] == 0.0)


        # print()
        # print()
        # print("x_lp", x_lp)
        # print()

        # print()
        # r, phi = x0_sph2.T
        # print("x0_sph2", r, phi/np.pi)
        # print()

#        self.lpnCoords.matrix_conversion_lpn_xyz(*self.lpnCoords.unit_vectors_lpn(x0_xyz, k0_xyz))


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

        if not isinstance(x0_xyz,np.ndarray):
            x0_xyz = np.array(x0_xyz)

        if not isinstance(k0_xyz,np.ndarray):
            k0_xyz = np.array(k0_xyz)


        # CONVERT TO LPN COORDINATES
        


        # x0_lp, l_, p_ = self.lpnCoords.impact_vector_no_vec(k0_xyz, x0_xyz)
        # print("x0_lp", x0_lp)

        # # NEED TO CHECK FOR SINGULAR TRANSFORMATIONS!!!
        # if np.any((x0_lp.T)[1] == 0.0): 
        #     print("SINGULAR", (x0_lp.T)[1] == 0.0)
        # else:
        l_hat, p_hat, n_hat = self.lpnCoords.unit_vectors_lpn_no_vec(k0_xyz, x0_xyz)
        M_xyz_lpn, M_lpn_xyz = self.lpnCoords.matrix_conversion_lpn_xyz_no_vec(l_hat, p_hat, n_hat)

        x0_lpn = M_xyz_lpn@x0_xyz
        x0_lpn[2] = 0.
        k0_lpn = M_xyz_lpn@k0_xyz
        # print("x0_xyz", x0_xyz)
        # print("x0_lpn matrix", x0_lpn)
        # print("x0_xyz matrix back", M_lpn_xyz@x0_lpn)

        # print("k0_xyz", k0_xyz)
        # print("k0_lpn matrix", k0_lpn)
        # print("k0_xyz matrix back", M_lpn_xyz@k0_lpn)


        # CONVERT TO 2D SPHERICAL COORDINATES
        x0_sph, k0_sph = self.conversions.convert_xyz_to_sph(x0_lpn, k0_lpn)#x0_xyz, k0_xyz)
        
        # k0_sph = [k0_sph[0], 0.0, k0_sph[2]]
        # x0_sph = [x0_sph[0], 1/2 * sp.pi, x0_sph[2]]

        k_r_0, k_th_0, k_ph_0 = k0_sph
        r0, th0, ph0 = x0_sph
        # th0=1/2*sp.pi
        # k_th_0 = 0.
        # Calculate from norm of starting condition
        t0 = 0
        k_t_0 = self.metric.k_t_from_norm_lamb(k_r_0, k_th_0, k_ph_0, t0, r0, th0, ph0)

        k_0 = np.array([k_t_0, *k0_sph])
        x_0 = np.array([t0, *x0_sph])

        # print(k_0, x_0)

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

        result = self.integrator.integrate(\
                        k_0, x_0, self.metric.get_dk, hit_blackhole, \
                        stop_integration = stop_integration,\
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

        # SHOULD I NOT CHANGE COORDS USING 4 VECTORS????
        x_lpn, k_lpn = self.conversions.convert_sph_to_xyz(x_sph, k_sph)
        
        k_xyz = np.array([M_lpn_xyz@k_i for k_i in k_lpn.T]).T
        x_xyz = np.array([M_lpn_xyz@x_i for x_i in x_lpn.T]).T
        
        x4_xyz = np.array([t, *x_xyz])
        k4_xyz = np.array([k_t, *k_xyz])

        result.update({"k4_xyz": k4_xyz, "x4_xyz": x4_xyz})

        return k_xyz, x_xyz, result


################################################################################################
################################################################################################
class SchwarzschildMetricSpherical2D:
################################################################################################
################################################################################################

    conversions4D = Conversions4D()

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, time_like = False, verbose=False):

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

        self.g__mu__nu = self.g__mu__nu.subs(self.th, 0.5*sp.pi)
        self.g_mu_nu = self.g__mu__nu.inv()
        self.g__mu__nu_diff = [self.g__mu__nu.diff(self.t), self.g__mu__nu.diff(self.r), \
                                     self.g__mu__nu.diff(self.th), self.g__mu__nu.diff(self.ph)]

        # We lambdify these to get numpy arrays
        self.g__mu__nu_lamb = sp.lambdify([self.t, self.r, self.th, self.ph], self.g__mu__nu, "numpy")
        self.g_mu_nu_lamb = sp.lambdify([self.t, self.r, self.ph], self.g_mu_nu, "numpy")
        self.g__mu__nu_diff_lamb = sp.lambdify([self.t, self.r, self.ph], self.g__mu__nu_diff, "numpy")

        # We integrate in the plane th=1/2*pi
        # self.g_mu_nu = self.g_mu_nu.subs(self.th, 0.5*sp.pi)


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

################################################################################################
################################################################################################
class LPN_coordinates:
################################################################################################
################################################################################################

    ### VECTORIZED LIN ALG FUNCTIONS ###

    def lp_to_sph2(self, x_lp):
        # k_lp = [1, 0]
        l, p = x_lp.T
        phi = np.atan2(p, l)
        r = np.sqrt(l**2 + p**2)
        x_sph2 = np.array([r, phi]).T

        dr_l = l/r
        dr_p = p/r
        dphi_l = -p/r**2
        dphi_p = -l/r**2

        # Since k_lp = [1, 0]
        k_sph2 = np.array([dr_l, dphi_l]).T

        return k_sph2, x_sph2


    def sph2_to_lp(self, k_sph2, x_sph2):
        r, phi = x_sph2.T
        l = np.cos(phi) * r
        p = np.sin(phi) * r

        dl_r = l/r
        dl_phi = -p
        dp_r = p/r
        dp_phi = l

        M = np.array([[dl_r, dl_phi], [dp_r, dp_phi]])

        x_lp = np.array([l, p]).T
        k_lp = np.array([M@k for k in k_sph2.T]) # BEter Doen!!

        return k_lp, x_lp


    def impact_vector_no_vec(self, k, x):
        k = np.array([k])
        x = np.array([x])
        _, l_, p_ = self.impact_vector(k, x)
        l, p = _.T
        return np.array([l[0], p[0]]).T, l_[0], p_[0]

    def impact_vector(self, k, x):
        """Vectorized calculating if impact vectors and distance vectors based on location and direction"""

        # print(k)
        # print(np.linalg.norm(k, axis=1))
        nk = 1./np.linalg.norm(k, axis=1)
        k = (k.T*nk).T#, axis=1)

        l = np.einsum('ij,ij->i', x, k) # Row-wise inner product of vectors
        #l_ = l * k
        l_ = (l * k.T).T
        
        p_ = x - l_
        p = np.linalg.norm(p_, axis=1)

        return np.array([l, p]).T, l_, p_

    def unit_vectors_lpn_no_vec(self, k, x):
        k = np.array([k])
        x = np.array([x])
        l_hat, p_hat, n_hat = self.unit_vectors_lpn(k, x)
        return l_hat[0], p_hat[0], n_hat[0]

    def unit_vectors_lpn(self, k, x):
        l_ = k
        n_ = np.cross(l_, x)
        # for i, in_ in enumerate(n_):
        #     print(in_, np.linalg.norm(in_), l_[i])
        #     if np.linalg.norm(in_) == 0.0: # Singular!
        #         print("Fixing")
        #         ra = np.random.rand(3)
        #         n_[i] = np.cross(l_[i], ra) # WAAROM WERKT DEZE ASSIGNMENT NIET????
        #         print(n_[i], ra, l_[i], np.cross(l_[i], ra))
        # print(n_)
        p_ = np.cross(n_, l_)

        norm_l = np.linalg.norm(l_, axis=1)
        norm_n = np.linalg.norm(n_, axis=1)
        norm_p = np.linalg.norm(p_, axis=1)

        l_hat = (1./norm_l * l_.T).T
        p_hat = (1./norm_p * p_.T).T
        n_hat = (1./norm_n * n_.T).T

        return l_hat, p_hat, n_hat

    def matrix_conversion_lpn_xyz_no_vec(self, l_hat, p_hat, n_hat):
        l_hat = np.array([l_hat])
        p_hat = np.array([p_hat])
        n_hat = np.array([n_hat])

        M_xyz_lpn, M_lpn_xyz = self.matrix_conversion_lpn_xyz(l_hat, p_hat, n_hat)

        return M_xyz_lpn[0], M_lpn_xyz[0]

    def matrix_conversion_lpn_xyz(self, l_hat, p_hat, n_hat):
        # Can this be faster without list comprehension?

        # M_xyz_lpn = np.array([l_hat[0], p_hat[0], n_hat[0]]).T
        M_lpn_xyz = [np.array([l_hat[i], p_hat[i], n_hat[i]]).T for i in range(l_hat.shape[0])]
        M_xyz_lpn = [np.linalg.inv(M) for M in M_lpn_xyz]
        return M_xyz_lpn, M_lpn_xyz#np.linalg.inv(M_xyz_lpn)


    # def unit_vectors_lpn_2(self, x, k):
    #     x_lp, l_ , p_ = impact_vector_2(x, k)
    #     p, l = x_lp.T

    #     p_hat = p_/np.linalg.norm(p_)
    #     l_hat = l_/np.linalg.norm(l_)
    #     n_hat = np.cross(l_hat, p_hat)

    #     return l_hat, p_hat, n_hat

    # def impact_vector(self, x, k):
    #     l_hat, p_hat, n_hat = unit_vectors_lpn(x, k)
    #     M_xyz_lpn, M_lpn_xyz = matrix_conversion_lpn_xyz(l_hat, p_hat, n_hat)

    #     l, p, n = np.array([M_xyz_lpn@xi for xi in x]).T
    #     p_ = (p*p_hat.T).T
    #     l_ = (l*l_hat.T).T

    #     return p, l, p_, l_


    # Angle between two 3-vectors
    def angle(self, v1, v2):
        """Return the angle between two 3-vectors. Vectorized."""
        arg = np.einsum('ij,ij->i', v1, v2) # Row-wise inner product of vectors
        norm = np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)
        #arg = np.dot(v1, v2)/ (np.linalg.norm(v1)*np.linalg.norm(v2))
        arg = (1/norm * arg.T).T
      
        return np.arccos(arg)

    # Create a quaternion for rotating a 3-vector in a plane (with normal n) by
    # an angle theta
    # q = create_quat(dTh, np.cross(p_, l_))
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
    def create_quat(self, theta, n):
        n = n / np.linalg.norm(n)
        w = np.cos(theta / 2)
        xyz = np.sin(theta/2) * n
        return R.from_quat(np.append(xyz, w))




