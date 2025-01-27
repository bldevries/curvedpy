import unittest
import curvedpy as cp
import sympy as sp
import numpy as np
import random
from curvedpy.utils.conversions import Conversions
from curvedpy.geodesics.kerr import GeodesicIntegratorKerr



# python -m unittest discover -v test
# python test/test_kerr_geodesic_integration.py TestCurvedpyKERR_conservation.test_check_conserved_quantities_photons

class TestCurvedpyKerr(unittest.TestCase):

    def setUp(self):
        self.converter = Conversions()
        self.a = 0.5
        self.m = 0.5
        self.gi = GeodesicIntegratorKerr(mass= self.m, a = self.a)#metric='schwarzschild', mass=1.0)
        self.gi_rev = GeodesicIntegratorKerr(mass= self.m, a = -self.a)#metric='schwarzschild', mass=1.0)


        self.start_t, self.end_t, self.steps = 0, 60, 60
        self.max_step = 0.1
        self.round_level = 4

    def test_KERR_check_direction_symmetry(self):  
        # Check if you get the same line forward or backward

        # Line in forward direction
        k0_xyz = np.array([1, 0.0, 0.0])
        x0_xyz = np.array([-10, 10, 0])

        k_xyz, x_xyz, line = self.gi.calc_trajectory(k0_xyz, x0_xyz, \
                                  curve_start = self.start_t, \
                                  curve_end = self.end_t, \
                                  nr_points_curve = self.steps,\
                                 max_step=self.max_step)

        k_end = sp.Matrix([k_xyz[0][-1], k_xyz[1][-1], k_xyz[2][-1]])

        # Rotation matrix for 180 degrees rotation around the z axis
        th = sp.Symbol("\\theta")
        R90 = sp.rot_axis3(th).subs(th, sp.pi)
        k_0_back = R90*k_end

        k0_xyz2 = np.array(k_0_back).astype(np.float64).flatten() #
        x0_xyz2 = np.array([x_xyz[0][-1], x_xyz[1][-1], x_xyz[2][-1]])


        # Trajectory in backward direction, but for the kerr metric we then need
        # to change the rotation direction of the blachhole
        k_xyz2, x_xyz2, line_reverse = self.gi_rev.calc_trajectory(k0_xyz2, x0_xyz2, \
                                  curve_start = self.start_t, \
                                  curve_end = self.end_t, \
                                  nr_points_curve = self.steps,\
                                 max_step=self.max_step)

        self.assertTrue( bool((np.round(x_xyz[0],self.round_level) == np.round(np.flip(x_xyz2[0]),self.round_level)).all()) )
        self.assertTrue( bool((np.round(x_xyz[1],self.round_level) == np.round(np.flip(x_xyz2[1]),self.round_level)).all()) )
        self.assertTrue( bool((np.round(x_xyz[2],self.round_level) == np.round(np.flip(x_xyz2[2]),self.round_level)).all()) )




    # def test_check_constant_kt(self):
    #     k0_xyz = np.array([1, 0.0, 0.0])
    #     x0_xyz = np.array([-10, 10, 0])

    #     k_xyz, x_xyz, line = self.gi.calc_trajectory(k0_xyz, x0_xyz, \
    #                               curve_start = self.start_t, \
    #                               curve_end = self.end_t, \
    #                               nr_points_curve = self.steps,\
    #                              max_step=self.max_step)
    
    #     k_r, r, k_th, th, k_ph, ph, k_t = line.y
    #     k_t = self.gi.k_t_from_norm_lamb(k_r, r, k_th, th, k_ph, ph, self.gi.r_s_value)

    #     self.assertTrue( bool(np.std(k_t) < 0.077) ) # Could be something to improve?

    #     # Also check if the norm of a null ray in nicely zero
    #     norm_k = self.gi.norm_k_lamb(k_t, k_r, r, k_th, th, k_ph, ph, self.gi.r_s_value)
    #     self.assertTrue( round(np.std(norm_k),8) == 0.0 )


################################################################################################
# Testing if the energy and angular momentum is conserved for orbits of photons or massive particles.
# This test is similar to what Bronzwaer et al. 18 and Bronzwaer et al. 21
################################################################################################
class TestCurvedpyKERR_conservation(unittest.TestCase):
    def setUp(self):
        self.converter = Conversions()
        self.a = 0.5
        self.m = 0.5
        self.gi = GeodesicIntegratorKerr(mass= self.m, a = self.a)
        self.gi_mass = GeodesicIntegratorKerr(mass= self.m, a = self.a, time_like = True)

        self.start_t, self.end_t, self.steps = 0, 60, 60
        self.max_step = 0.1
        self.round_level = 11

    def test_KERR_check_conserved_quantities_photons(self):
        k0_xyz=np.array([1, 0, 1])
        x0_xyz=np.array([0.000001, 10, 0])
        k, x, res = self.gi.calc_trajectory(k0_xyz, x0_xyz,\
                                    curve_start = self.start_t, \
                                    curve_end = self.end_t, \
                                    nr_points_curve = self.steps,\
                                    max_step=self.max_step)

        k_r, r, k_th, th, k_ph, ph, k_t = res.y
        k, x = [k_t, k_r, k_th, k_ph], [res.t, r, th, ph]

        k__ = self.gi.lamb_k__(*k, *x, self.gi.r_s_value, self.gi.a_value)
        k__= k__.reshape(4,k__.shape[-1])
        
        #np.std(k__[0]), np.std(k__[1]), np.std(k__[2]), np.std(k__[3])
        self.assertTrue( round(np.std(k__[0]),self.round_level) == 0.0 )
        self.assertTrue( round(np.std(k__[3]),self.round_level) == 0.0 )

    def test_KERR_check_conserved_quantities_massive(self):
        k0_xyz=np.array([1, 0, 1])
        x0_xyz=np.array([0.000001, 10, 0])
        k, x, res = self.gi_mass.calc_trajectory(k0_xyz, x0_xyz,\
                                    curve_start = self.start_t, \
                                    curve_end = self.end_t, \
                                    nr_points_curve = self.steps,\
                                    max_step=self.max_step)

        k_r, r, k_th, th, k_ph, ph, k_t = res.y
        k, x = [k_t, k_r, k_th, k_ph], [res.t, r, th, ph]

        k__ = self.gi.lamb_k__(*k, *x, self.gi.r_s_value, self.gi.a_value)
        k__= k__.reshape(4,k__.shape[-1])
        
        #np.std(k__[0]), np.std(k__[1]), np.std(k__[2]), np.std(k__[3])
        self.assertTrue( round(np.std(k__[0]),self.round_level) == 0.0 )
        self.assertTrue( round(np.std(k__[3]),self.round_level) == 0.0 )



if __name__ == '__main__':
    unittest.main()