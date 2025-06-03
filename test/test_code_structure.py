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


    def test_basics_integrator_info(self):
        self.assertTrue( self.gi_ss.get_m() == self.mass )
        self.assertTrue( self.gi_ss.get_a() == 0.0 )
        self.assertTrue( self.gi_kerr.get_m() == self.mass )
        self.assertTrue( self.gi_kerr.get_a() == self.a )


if __name__ == '__main__':
    unittest.main()