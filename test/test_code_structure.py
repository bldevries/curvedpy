import unittest
import curvedpy as cp
import sympy as sp
import numpy as np
import random
from curvedpy.utils.conversions import Conversions
from curvedpy.geodesics.blackhole import BlackholeGeodesicIntegrator
# from curvedpy.geodesics.blackhole_integrators.schwarzschild.schwarzschild_metric import SchwarzschildMetricSpherical
# from curvedpy.utils.conversions_SPH2PATCH_3D import cart_to_sph1

# python -m unittest discover -v test
# python test/test_geodesic_integration.py -v


class TestBasics(unittest.TestCase):
    def setUp(self):
        self.converter = Conversions()
        self.mass = 2.3
        self.a = 1.3
        self.gi_ss = BlackholeGeodesicIntegrator(mass=self.mass)
        self.gi_kerr = BlackholeGeodesicIntegrator(mass=self.mass, a = self.a)


    # Simple test to see if the mass and spin are set correctly
    def test_basics_integrator_info(self):
        self.assertTrue( self.gi_ss.get_m() == self.mass )
        self.assertTrue( self.gi_ss.get_a() == 0.0 )
        self.assertTrue( self.gi_kerr.get_m() == self.mass )
        self.assertTrue( self.gi_kerr.get_a() == self.a )

    # Testing if all comes out well if calculating multiple geodesics
    def test_vectorized_geodesics(self):
        k0 = np.array([1, 0, 0])
        x0 = np.array([-15, 15, 10])
        nr_points_curve = 57

        # SCHWARZ
        _, x, _ = self.gi_ss.geodesic(k0, x0, nr_points_curve=nr_points_curve)
        self.assertTrue( x.shape[0] == k0.shape[0] )
        self.assertTrue( x.shape[1] == nr_points_curve )


        res = self.gi_ss.geodesic(\
                        np.array([k0 for i in range(10)]), \
                        np.array([x0 for i in range(10)]), \
                        nr_points_curve=nr_points_curve)

        self.assertTrue( len(res) == 10 )
        _, x, _ = res[0]
        self.assertTrue( x.shape[0] == k0.shape[0] )
        self.assertTrue( x.shape[1] == nr_points_curve )

        # KERR
        _, x, _ = self.gi_kerr.geodesic(k0, x0, nr_points_curve=nr_points_curve)
        self.assertTrue( x.shape[0] == k0.shape[0] )
        self.assertTrue( x.shape[1] == nr_points_curve )


        res = self.gi_kerr.geodesic(\
                        np.array([k0 for i in range(10)]), \
                        np.array([x0 for i in range(10)]), \
                        nr_points_curve=nr_points_curve)

        self.assertTrue( len(res) == 10 )
        _, x, _ = res[0]
        self.assertTrue( x.shape[0] == k0.shape[0] )
        self.assertTrue( x.shape[1] == nr_points_curve )


    # Simple test to see if R_end is implemented
    def test_R_end(self):
        R_end = 20
        k0 = [1, 0, 0]
        x0 = [-15, 15, 10]

        # SCHWARZ
        _, x_no_end, _ = self.gi_ss.geodesic(k0, x0)
        _, x, _ = self.gi_ss.geodesic(k0, x0, R_end = R_end)
        self.assertTrue( np.linalg.norm(x_no_end.T[-1]) >=  np.linalg.norm(x.T[-1]) )

        # KERR
        k_no_end, x_no_end, _ = self.gi_kerr.geodesic(k0, x0)
        _, x, _ = self.gi_kerr.geodesic(k0, x0, R_end = R_end)
        self.assertTrue( np.linalg.norm(x_no_end.T[-1]) >=  np.linalg.norm(x.T[-1]) )


if __name__ == '__main__':
    unittest.main()