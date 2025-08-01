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
# python test/test_geodesic_integration.py -v


################################################################################################
# 
################################################################################################

class TestCurvedpySchwarzschildPointPoint(unittest.TestCase):

    def setUp(self):
        self.gi = BlackholeGeodesicPointPointIntegrator()
        self.max_step = 0.1
        self.round_level = 0


    def test_check_end_point_lpn(self):  
        x0_xyz = np.array([[0,10,5]])
        xf_xyz = np.array([[5,0,3]])
        M_xyz_lpn, M_lpn_xyz = self.gi.matrix_conversion_lpn_xyz(*self.gi.unit_vectors_lpn(x0_xyz, xf_xyz))

        x0_lpn = M_xyz_lpn[0]@x0_xyz[0]
        xf_lpn = M_xyz_lpn[0]@xf_xyz[0]

        k0_lpn, length, l_k_lpn, l_x_lpn, l_results = self.gi.geodesic_pp_lpn(xf_lpn, x0_lpn, max_step = self.max_step, eps_r=0.001, image_nr=1)
        x_xyz = M_lpn_xyz[0]@l_x_lpn[0]
        self.assertTrue( round( np.linalg.norm(xf_xyz[0] - x_xyz.T[-1]), self.round_level ) == 0.0 )

        k0_lpn, length, l_k_lpn, l_x_lpn, l_results = self.gi.geodesic_pp_lpn(x0_lpn, xf_lpn, max_step = self.max_step, eps_r=0.001, image_nr=1)
        x_xyz = M_lpn_xyz[0]@l_x_lpn[0]
        # I use x_xyz.T[0] and not x_xyz.T[-1] since the algorithm
        self.assertTrue( round( np.linalg.norm(x0_xyz[0] - x_xyz.T[-1]), self.round_level ) == 0.0 )

    def test_check_end_point(self):  
        x0_xyz = np.array([[0,10,5]])
        xf_xyz = np.array([[5,0,3]])

        x_xyz = self.gi.geodesic_pp(xf_xyz, x0_xyz, max_step = self.max_step, eps_r=0.001, image_nr=1)
        self.assertTrue( round( np.linalg.norm(xf_xyz[0] - x_xyz[0].T[-1]), self.round_level ) == 0.0 )

        x_xyz = self.gi.geodesic_pp(x0_xyz, xf_xyz, max_step = self.max_step, eps_r=0.001, image_nr=1)
        self.assertTrue( round( np.linalg.norm(x0_xyz[0] - x_xyz[0].T[-1]), self.round_level ) == 0.0 )


#     def test_SCHW_check_direction_symmetry(self):  
#         # Check if you get the same line forward or backward

#         # Line in forward direction
#         k0_xyz = np.array([1, 0.0, 0.0])
#         x0_xyz = np.array([-10, 10, 0])

#         k_xyz, x_xyz, line = self.gi.geodesic(k0_xyz, x0_xyz, \
#                                   curve_start = self.start_t, \
#                                   curve_end = self.end_t, \
#                                   nr_points_curve = self.steps,\
#                                  max_step=self.max_step)

#         k_end = sp.Matrix([k_xyz[0][-1], k_xyz[1][-1], k_xyz[2][-1]])

#         # Rotation matrix for 180 degrees rotation around the z axis
#         th = sp.Symbol("\\theta")
#         R90 = sp.rot_axis3(th).subs(th, sp.pi)
#         k_0_back = R90*k_end

#         k0_xyz2 = np.array(k_0_back).astype(np.float64).flatten() #
#         x0_xyz2 = np.array([x_xyz[0][-1], x_xyz[1][-1], x_xyz[2][-1]])


#         # Trajectory in backward direction
#         k_xyz2, x_xyz2, line_reverse = self.gi.geodesic(k0_xyz2, x0_xyz2, \
#                                   curve_start = self.start_t, \
#                                   curve_end = self.end_t, \
#                                   nr_points_curve = self.steps,\
#                                  max_step=self.max_step)

#         self.assertTrue( bool((np.round(x_xyz[0],self.round_level) == np.round(np.flip(x_xyz2[0]),self.round_level)).all()) )
#         self.assertTrue( bool((np.round(x_xyz[1],self.round_level) == np.round(np.flip(x_xyz2[1]),self.round_level)).all()) )
#         self.assertTrue( bool((np.round(x_xyz[2],self.round_level) == np.round(np.flip(x_xyz2[2]),self.round_level)).all()) )



# ################################################################################################
# # Testing if the energy and angular momentum is conserved for orbits of photons or massive particles.
# # This test is similar to what Bronzwaer et al. 18 and Bronzwaer et al. 21
# ################################################################################################
# class TestCurvedpySchwarzschild_conservation(unittest.TestCase):
#     def setUp(self):
#         self.converter = Conversions()

#         self.mass = 1.0
#         self.start_t, self.end_t, self.steps = 0, 60, 60
#         self.max_step = 0.1
#         self.round_level = 6

#         # self.metric_sph = SchwarzschildMetricSpherical(mass = self.mass)


#     def test_SCHW_SPH_check_conserved_quantities_photons(self):
#         self.gi = BlackholeGeodesicIntegrator(mass = self.mass, time_like = False)#coordinates="SPH2PATCH", 

#         k0_sph = np.array([0.0, 0., -0.1]) 
#         x0_sph = np.array([3, 1/2*np.pi, 1/4*np.pi])

#         x0_xyz, k0_xyz = self.converter.convert_sph_to_xyz(x0_sph, k0_sph)
#         k, x, res =  self.gi.geodesic(k0_xyz, x0_xyz, max_step=self.max_step)#curve_end = 100, nr_points_curve = 1000, 

#         k4 = res['k4_sph']
#         x4 = res['x4_sph']

#         x_th = x4[2]

#         k4__mu = self.gi.get_metric().oneform(k4, x4)
#         L = k4__mu[3]
#         E = k4__mu[0]

#         self.assertTrue( round(np.std(L),self.round_level) == 0.0 )
#         self.assertTrue( round(np.std(E),self.round_level) == 0.0 )

#         # Check if in SPH coordinates it stays in the th=1/2pi surface
#         self.assertTrue( round(np.std(x_th),self.round_level) == 0.0 )


#     def test_SCHW_SPH_check_conserved_quantities_photons_multiple(self):
#         self.gi = BlackholeGeodesicIntegrator(mass = self.mass, time_like = False)#coordinates="SPH2PATCH", 

#         k0_sph = np.array([[0.0, 0., -0.1] for i in range(10)])
#         x0_sph = np.array([[3, 1/2*np.pi, 1/4*np.pi] for i in range(10)])

#         x0_xyz, k0_xyz = self.converter.convert_sph_to_xyz(x0_sph, k0_sph, vec=True)

#         results =  self.gi.geodesic(k0_xyz, x0_xyz, max_step=self.max_step)#curve_end = 100, nr_points_curve = 1000, 


#         for item in results:
#             k, x, res = item

#             k4 = res['k4_sph']
#             x4 = res['x4_sph']

#             x_th = x4[2]

#             k4__mu = self.gi.get_metric().oneform(k4, x4)
#             L = k4__mu[3]
#             E = k4__mu[0]

#             self.assertTrue( round(np.std(L),self.round_level) == 0.0 )
#             self.assertTrue( round(np.std(E),self.round_level) == 0.0 )

#             # Check if in SPH coordinates it stays in the th=1/2pi surface
#             self.assertTrue( round(np.std(x_th),self.round_level) == 0.0 )


#     def test_SCHW_SPH_check_conserved_quantities_photons_mp(self):
#         self.gi = BlackholeGeodesicIntegrator(mass = self.mass, time_like = False)#coordinates="SPH2PATCH", 

#         k0_sph = np.array([[0.0, 0., -0.1] for i in range(33)])
#         x0_sph = np.array([[3, 1/2*np.pi, 1/4*np.pi] for i in range(33)])

#         x0_xyz, k0_xyz = self.converter.convert_sph_to_xyz(x0_sph, k0_sph, vec=True)

#         results =  self.gi.geodesic_mp(k0_xyz, x0_xyz, cores=4, split_factor = 4, max_step=self.max_step)#curve_end = 100, nr_points_curve = 1000, 

#         for item in results:
#             k, x, res = item

#             k4 = res['k4_sph']
#             x4 = res['x4_sph']

#             x_th = x4[2]

#             k4__mu = self.gi.get_metric().oneform(k4, x4)
#             L = k4__mu[3]
#             E = k4__mu[0]

#             self.assertTrue( round(np.std(L),self.round_level) == 0.0 )
#             self.assertTrue( round(np.std(E),self.round_level) == 0.0 )

#             # Check if in SPH coordinates it stays in the th=1/2pi surface
#             self.assertTrue( round(np.std(x_th),self.round_level) == 0.0 )

if __name__ == '__main__':
    unittest.main()