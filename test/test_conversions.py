import unittest
import curvedpy as cp
import sympy as sp
import numpy as np
import random
from curvedpy.utils.conversions import Conversions

################################################################################################
#
################################################################################################
class TestConversions(unittest.TestCase):
    def setUp(self):
        self.converter = Conversions()

    # A single test to see if the spherical and cartesian conversion works
    def test_CONV_sph_to_xyz_and_back(self):
        k0_xyz = np.array([11.322145, 15.136237, 65.246265])
        x0_xyz = np.array([13.461341, 13.461346, 72.300564])
        round_lvl = 6

        x0_sph, k0_sph = self.converter.convert_xyz_to_sph(x0_xyz, k0_xyz)
        x0_xyz_new, k0_xyz_new = self.converter.convert_sph_to_xyz(x0_sph, k0_sph)

        self.assertTrue( bool((k0_xyz == [round(v, round_lvl) for v in k0_xyz_new]).all()) )
        self.assertTrue( bool((x0_xyz == [round(v, round_lvl) for v in x0_xyz_new]).all()) )

    # Sample of tests to see if the conversion from Boyer-Lindquist to xyz and back conversion works
    def test_CONV_xyz_to_bl_and_back_conversions(self):
        round_lvl = 13
        xl, yl, zl = [], [], []
        kxl, kyl, kzl = [], [], []
        for i in range(100):
            x0_xyz = np.array([random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100)])
            k0_xyz = np.array([random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100)])
            
            x0, y0, z0 = x0_xyz
            kx0, ky0, kz0 = k0_xyz
            
            x_bl, k_bl = self.converter.convert_xyz_to_bl(x0_xyz, k0_xyz, a=1)
            
            x_xyz, k_xyz = self.converter.convert_bl_to_xyz(x_bl, k_bl, a=1)

            x, y, z = x_xyz
            kx, ky, kz = k_xyz
            
            xl.append(x-x0)
            yl.append(y-y0)
            zl.append(z-z0)
            kxl.append(kx-kx0)
            kyl.append(ky-ky0)
            kzl.append(kz-kz0)

        np.std(xl), np.std(yl), np.std(zl), np.std(kxl), np.std(kyl), np.std(kzl)
        self.assertTrue( round(np.std(xl), round_lvl) == 0.0)

if __name__ == '__main__':
    unittest.main()