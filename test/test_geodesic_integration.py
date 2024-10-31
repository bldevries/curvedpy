import unittest
import curvedpy as cp
import sympy as sp
import numpy as np

# python -m unittest discover -v test

class TestCurvedpyFlatMetric(unittest.TestCase):

    def setUp(self):
        self.gi = cp.GeodesicIntegrator(metric='flat')
        start_t, end_t, steps = 0, 60, 60
        self.line = self.gi.calc_trajectory(k_x_0 = 1., x0 = 10, k_y_0 = 0., y0 = 0, k_z_0 = 0., z0 = 0.0, \
                                  curve_start = start_t, \
                                  curve_end = end_t, \
                                  nr_points_curve = steps)
        self.k_x, self.x, self.k_y, self.y, self.k_z, self.z = self.line.y
        self.t = self.line.t

    def test_straight_line(self):
        self.assertEqual(self.y[0] , self.y[-1])
        self.assertEqual(self.z[0] , self.z[-1])
        #x[0], x[-1], bool(y[0] == y[-1]), z[0] == z[-1], len(x) == steps, len(t), t[0], t[-1]

class TestCurvedpySchwarzschild(unittest.TestCase):

    def setUp(self):
        self.gi = cp.GeodesicIntegrator(metric='schwarzschild', mass=1.0)
        self.start_t, self.end_t, self.steps = 0, 60, 60
        self.max_step = 1
        self.round_level = 4

    def test_check_direction_symmetry(self):  
        # Check if you get the same line forward or backward

        # Line in forward direction
        line = self.gi.calc_trajectory(k_x_0 = 1., x0 = -10, k_y_0 = 0., y0 = 5, k_z_0 = 0., z0 = 0.0, \
                                  curve_start = self.start_t, \
                                  curve_end = self.end_t, \
                                  nr_points_curve = self.steps,\
                                 max_step=self.max_step)#,\
        k_x, x, k_y, y, k_z, z = line.y

        x_end = sp.Matrix([x[-1], y[-1], z[-1]])
        k_end = sp.Matrix([k_x[-1], k_y[-1], k_z[-1]])

        # Rotation matrix for 180 degrees rotation around the z axis
        th = sp.Symbol("\\theta")
        R90 = sp.rot_axis3(th).subs(th, sp.pi)
        k_0_back = R90*k_end

        # Trajectory in backward direction
        line_reverse = self.gi.calc_trajectory(k_x_0 = float(k_0_back[0]), x0 = float(x_end[0]), k_y_0 = float(k_0_back[1]), \
                                  y0 = float(x_end[1]), k_z_0 = float(k_0_back[2]), z0 = float(x_end[2]), \
                                  curve_start = self.start_t, \
                                  curve_end = self.end_t, \
                                  nr_points_curve = self.steps,\
                                 R_end= 40, max_step=self.max_step)
        k_x2, x2, k_y2, y2, k_z2, z2 = line_reverse.y

        self.assertTrue( bool((np.round(x,self.round_level) == np.round(np.flip(x2),self.round_level)).all()) )


    def test_check_constant_kt(self):
        line = self.gi.calc_trajectory(k_x_0 = 1., x0 = -10, k_y_0 = 0., y0 = 5, k_z_0 = 0., z0 = 0.0, \
                                  curve_start = self.start_t, \
                                  curve_end = self.end_t, \
                                  nr_points_curve = self.steps,\
                                 max_step=self.max_step)#,\
    
        k_t = self.gi.k_t_from_norm_lamb(*line.y, self.gi.r_s_value)

        self.assertTrue( bool(np.std(k_t) < 0.3) ) # THIS NEEDS IMPROVEMENT!!

        # Also check if the norm of a null ray in nicely zero
        norm_k = self.gi.norm_k_lamb(k_t, *line.y, self.gi.r_s_value)
        self.assertTrue( round(np.std(norm_k),8) == 0.0 )




if __name__ == '__main__':
    unittest.main()