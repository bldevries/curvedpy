import unittest
import curvedpy as cp
import sympy as sp
import numpy as np
import random

# python -m unittest discover -v test

################################################################################################
# Test to see if the spherical to cartesian conversion functions are consistent
################################################################################################
class TestConversions(unittest.TestCase):
    def setUp(self):
        self.converter = cp.Conversions()

    def test_sph_to_xyz_and_back(self):
        k0_xyz = np.array([11.322145, 15.136237, 65.246265])
        x0_xyz = np.array([13.461341, 13.461346, 72.300564])
        round_lvl = 6

        x0_sph, k0_sph = self.converter.convert_xyz_to_sph(x0_xyz, k0_xyz)
        x0_xyz_new, k0_xyz_new = self.converter.convert_sph_to_xyz(x0_sph, k0_sph)

        self.assertTrue( bool((k0_xyz == [round(v, round_lvl) for v in k0_xyz_new]).all()) )
        self.assertTrue( bool((x0_xyz == [round(v, round_lvl) for v in x0_xyz_new]).all()) )


################################################################################################
# Test to check if the geodesic is the same in the other direction. This means the 
# metric is not time dependent.
################################################################################################

class TestCurvedpySchwarzschild(unittest.TestCase):

    def setUp(self):
        self.gi = cp.GeodesicIntegratorSchwarzschild()#metric='schwarzschild', mass=1.0)
        self.start_t, self.end_t, self.steps = 0, 60, 60
        self.max_step = 1
        self.round_level = 4

    def test_check_direction_symmetry(self):  
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


        # Trajectory in backward direction
        k_xyz2, x_xyz2, line_reverse = self.gi.calc_trajectory(k0_xyz2, x0_xyz2, \
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
class TestCurvedpySchwarzschild_conservation(unittest.TestCase):
    def setUp(self):
        self.converter = cp.Conversions()

        self.mass = 1.0
        self.start_t, self.end_t, self.steps = 0, 60, 60
        self.max_step = 0.1
        self.round_level = 10

    def test_check_conserved_quantities_photons(self):
        self.gi = cp.GeodesicIntegratorSchwarzschild(mass = self.mass, time_like = False)#metric='schwarzschild', mass=1.0)

        k0_sph = np.array([0.0, 0., -0.1]) 
        x0_sph = np.array([3, 1/2*np.pi, 1/4*np.pi])
        x0_xyz, k0_xyz = self.converter.convert_sph_to_xyz(x0_sph, k0_sph)
        k, x, res =  self.gi.calc_trajectory(k0_xyz, x0_xyz, verbose=False, max_step=0.01)#curve_end = 100, nr_points_curve = 1000, 
        k_r, r, k_th, th, k_ph, ph, k_t = res.y

        L = self.gi.ang_mom(r, k_ph)
        self.assertTrue( round(np.mean(L),self.round_level) == round(self.gi.ang_mom(x0_sph[0], k0_sph[2]),self.round_level) )
        self.assertTrue( round(np.std(L),self.round_level) == 0.0 )

        E = self.gi.energy_photon(k_r, r, k_ph, M_blackhole=1)
        self.assertTrue( round(np.mean(E),self.round_level) == round(self.gi.energy_photon(k_r = k0_sph[0], r = x0_sph[0], k_ph = k0_sph[2], M_blackhole=self.mass),self.round_level) )
        self.assertTrue( round(np.std(E),self.round_level) == 0.0 )

    def test_check_conserved_quantities_massive_particles(self):
        self.gi = cp.GeodesicIntegratorSchwarzschild(mass = self.mass, time_like = True)#metric='schwarzschild', mass=1.0)

        k0_sph = np.array([0., 0., -0.1])
        x0_sph = np.array([20, 1/2*np.pi,0])
        x0_xyz, k0_xyz = self.converter.convert_sph_to_xyz(x0_sph, k0_sph)
        k, x, res =  self.gi.calc_trajectory(k0_xyz, x0_xyz, verbose=False, max_step=0.01)#curve_end = 100, nr_points_curve = 1000, 
        k_r, r, k_th, th, k_ph, ph, k_t = res.y

        L = self.gi.ang_mom(r, k_ph)
        self.assertTrue( round(np.mean(L),self.round_level) == round(self.gi.ang_mom(x0_sph[0], k0_sph[2]),self.round_level) )
        self.assertTrue( round(np.std(L),self.round_level) == 0.0 )

        E = self.gi.energy_massive(k_r, r, k_ph, M_blackhole=1)
        self.assertTrue( round(np.mean(E),self.round_level) == round(self.gi.energy_massive(k_r = k0_sph[0], r = x0_sph[0], k_ph = k0_sph[2], M_blackhole=self.mass),self.round_level) )
        self.assertTrue( round(np.std(E),self.round_level) == 0.0 )

if __name__ == '__main__':
    unittest.main()