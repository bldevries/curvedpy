import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
from curvedpy.utils.conversions import Conversions
from curvedpy.geodesics.blackhole_integrators.schwarzschild_XYZ import GeodesicIntegratorSchwarzschildXYZ

class BlackholeGeodesicIntegrator:

    conversions = Conversions()

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, a = 0, time_like = False, verbose=False):
        if a == 0:
            self.gi = GeodesicIntegratorSchwarzschildXYZ(mass=mass, time_like=time_like, verbose = verbose)
        else:
            print("Kerr blackhole not officially implemented. But you can try curvedpy.geodesics.blackhole.kerr_SPH")

    ################################################################################################
    #
    ################################################################################################
    def geodesic(self, k0_xyz, x0_xyz, *args, **kargs):
        return self.gi.calc_trajectory(k0_xyz, x0_xyz, *args, **kargs)


    ################################################################################################
    #
    ################################################################################################
    def get_r_s(self):
        return self.gi.r_s_value