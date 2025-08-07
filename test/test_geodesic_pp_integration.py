import unittest
import curvedpy as cp
import sympy as sp
import numpy as np
import random
from curvedpy.utils.conversions import Conversions
from curvedpy.geodesics.blackhole import BlackholeGeodesicIntegrator
from curvedpy.geodesics.blackhole_integrators.schwarzschild.schwarzschild_geodesic import IntegratorSchwarzschildSPH2D
from curvedpy.geodesics.pointpoint import BlackholeGeodesicPointPointIntegrator

# from curvedpy.geodesics.blackhole_integrators.schwarzschild.schwarzschild_metric import SchwarzschildMetricSpherical
# from curvedpy.utils.conversions_SPH2PATCH_3D import cart_to_sph1

# python -m unittest discover -v test
# python test/test_geodesic_pp_integration.py -v


################################################################################################
# 
################################################################################################

class TestCurvedpySchwarzschildPointPoint(unittest.TestCase):

    def setUp(self):
        self.gi = BlackholeGeodesicPointPointIntegrator()
        self.max_step = 0.1
        self.round_level = 0


    def test_check_geodesic_pp_lpn_no_vec(self):  
        x0_xyz = np.array([[0,10,5]])
        xf_xyz = np.array([[5,0,3]])
        M_xyz_lpn, M_lpn_xyz = self.gi.matrix_conversion_lpn_xyz(*self.gi.unit_vectors_lpn(x0_xyz, xf_xyz))

        x0_lpn = M_xyz_lpn[0]@x0_xyz[0]
        xf_lpn = M_xyz_lpn[0]@xf_xyz[0]

        translation_lpn, k0_lpn, length, l_k_lpn, l_x_lpn, l_results = self.gi.geodesic_pp_lpn_no_vec(xf_lpn, x0_lpn, max_step = self.max_step, eps_r=0.001, image_nr=1)
        x_xyz = M_lpn_xyz[0]@l_x_lpn
        self.assertTrue( round( np.linalg.norm(xf_xyz[0] - x_xyz.T[-1]), self.round_level ) == 0.0 )

        translation_lpn, k0_lpn, length, l_k_lpn, l_x_lpn, l_results = self.gi.geodesic_pp_lpn_no_vec(x0_lpn, xf_lpn, max_step = self.max_step, eps_r=0.001, image_nr=1)
        x_xyz = M_lpn_xyz[0]@l_x_lpn
        # I use x_xyz.T[0] and not x_xyz.T[-1] since the algorithm
        self.assertTrue( round( np.linalg.norm(x0_xyz[0] - x_xyz.T[-1]), self.round_level ) == 0.0 )

    def test_check_geodesic_pp(self):  
        x0_xyz = np.array([[0,10,5]])
        xf_xyz = np.array([[5,0,3]])

        x_xyz, l_translation_xyz = self.gi.geodesic_pp(xf_xyz, x0_xyz, max_step = self.max_step, eps_r=0.001, image_nr=1)
        self.assertTrue( round( np.linalg.norm(xf_xyz[0] - x_xyz[0].T[-1]), self.round_level ) == 0.0 )

        x_xyz, l_translation_xyz = self.gi.geodesic_pp(x0_xyz, xf_xyz, max_step = self.max_step, eps_r=0.001, image_nr=1)
        self.assertTrue( round( np.linalg.norm(x0_xyz[0] - x_xyz[0].T[-1]), self.round_level ) == 0.0 )


    def test_check_geodesic_pp_vectorization(self):  
        x0_xyz = np.array([[0,10,5], [0,16,3]])
        xf_xyz = np.array([[5,0,3], [3,1,9]])

        x_xyz, l_translation_xyz = self.gi.geodesic_pp(xf_xyz, x0_xyz, max_step = self.max_step, eps_r=0.001, image_nr=1)
        for i in range(len(x0_xyz)):
            self.assertTrue( round( np.linalg.norm(xf_xyz[i] - x_xyz[i].T[-1]), self.round_level ) == 0.0 )

    def test_check_geodesic_pp_image_nr(self):  
        x0_xyz = np.array([[0,16,3]])
        xf_xyz = np.array([[3,1,9]])

        for image_nr in [1,2,3]:
            x_xyz, l_translation_xyz = self.gi.geodesic_pp(xf_xyz, x0_xyz, max_step = self.max_step, \
                                                            eps_r=0.001, image_nr=image_nr)
            for i in range(len(x0_xyz)):
                self.assertTrue( round( np.linalg.norm(xf_xyz[i] - x_xyz[i].T[-1]), self.round_level ) == 0.0 )


if __name__ == '__main__':
    unittest.main()