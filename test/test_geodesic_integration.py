import unittest
import curvedpy as cp
import sympy as sp
import numpy as np

# python -m unittest discover -v test

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



class TestCurvedpySchwarzschild(unittest.TestCase):

    def setUp(self):
        self.gi = cp.GeodesicIntegratorSchwarzschild()#metric='schwarzschild', mass=1.0)
        self.start_t, self.end_t, self.steps = 0, 60, 60
        self.max_step = 1
        self.round_level = 4

    def test_check_direction_symmetry(self):  
        # Check if you get the same line forward or backward

        # Line in forward direction
        k0_xyz = [1, 0.0, 0.0]
        x0_xyz = [-10, 10, 0]

        k_xyz, x_xyz, line = self.gi.calc_trajectory(k0_xyz, x0_xyz, \
                                  curve_start = self.start_t, \
                                  curve_end = self.end_t, \
                                  nr_points_curve = self.steps,\
                                 max_step=self.max_step)#,\

        k_end = sp.Matrix([k_xyz[0][-1], k_xyz[1][-1], k_xyz[2][-1]])

        # Rotation matrix for 180 degrees rotation around the z axis
        th = sp.Symbol("\\theta")
        R90 = sp.rot_axis3(th).subs(th, sp.pi)
        k_0_back = R90*k_end

        k0_xyz2 = np.array(k_0_back).astype(np.float64).flatten() #
        x0_xyz2 = [x_xyz[0][-1], x_xyz[1][-1], x_xyz[2][-1]]


        # Trajectory in backward direction
        k_xyz2, x_xyz2, line_reverse = self.gi.calc_trajectory(k0_xyz2, x0_xyz2, \
                                  curve_start = self.start_t, \
                                  curve_end = self.end_t, \
                                  nr_points_curve = self.steps,\
                                 max_step=self.max_step)#R_end= 40, 

        self.assertTrue( bool((np.round(x_xyz[0],self.round_level) == np.round(np.flip(x_xyz2[0]),self.round_level)).all()) )
        self.assertTrue( bool((np.round(x_xyz[1],self.round_level) == np.round(np.flip(x_xyz2[1]),self.round_level)).all()) )
        self.assertTrue( bool((np.round(x_xyz[2],self.round_level) == np.round(np.flip(x_xyz2[2]),self.round_level)).all()) )




    def test_check_constant_kt(self):
        k0_xyz = [1, 0.0, 0.0]
        x0_xyz = [-10, 10, 0]
        k_xyz, x_xyz, line = self.gi.calc_trajectory(k0_xyz, x0_xyz, \
                                  curve_start = self.start_t, \
                                  curve_end = self.end_t, \
                                  nr_points_curve = self.steps,\
                                 max_step=self.max_step)#,\
    
        k_t = self.gi.k_t_from_norm_lamb(*line.y, self.gi.r_s_value)

        self.assertTrue( bool(np.std(k_t) < 0.077) ) # Could be something to improve?

        # Also check if the norm of a null ray in nicely zero
        norm_k = self.gi.norm_k_lamb(k_t, *line.y, self.gi.r_s_value)
        self.assertTrue( round(np.std(norm_k),8) == 0.0 )




if __name__ == '__main__':
    unittest.main()