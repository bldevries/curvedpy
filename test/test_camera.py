import unittest
import curvedpy as cp
import sympy as sp
import numpy as np
import random
from curvedpy.utils.conversions import Conversions
from curvedpy.cameras.camera import RelativisticCamera


# python -m unittest discover -v test
# python test/test_geodesic_integration.py -v

################################################################################################
#
################################################################################################
class TestCameras(unittest.TestCase):
    def setUp(self):
        self.converter = Conversions()


    def test_CAM_make_camera(self):
        self.camera = RelativisticCamera(resolution = [2, 2], verbose_init = False)
        self.camera.run_mp(cores=1)

        self.assertTrue( len(self.camera.results) == 4 )
        self.assertTrue( self.camera.ray_blackhole_hit.shape == (2,2) )
        self.assertTrue( self.camera.ray_end.shape == (2,2,6) ) 

if __name__ == '__main__':
    unittest.main()

