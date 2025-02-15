import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
from curvedpy.utils.conversions import Conversions
from curvedpy.geodesics.blackhole_integrators.schwarzschild_XYZ import GeodesicIntegratorSchwarzschildXYZ
from curvedpy.geodesics.blackhole_integrators.kerrschild_XYZ import GeodesicIntegratorKerrSchildXYZ
from curvedpy.geodesics.blackhole_integrators.schwarzschild_SPH1SPH2 import GeodesicIntegratorSchwarzschildSPH1SPH2


class BlackholeGeodesicIntegrator:

    conversions = Conversions()

    FORCE_COORDINATES_CARTESIAN = {"force_coordinates": "xyz", "Explanation": ""}
    FORCE_COORDINATES_SPH2PATCH = {"force_coordinates": "SPH2PATCH", "Explanation": ""}
    FORCE_COORDINATES_SPH = {"force_coordinates": "SPH", "Explanation": "", "Note": "Only for debugging!"}

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, a = 0, coordinates = "SPH2PATCH", theta_switch = 0.2*np.pi, time_like = False, auto_grad = False, verbose=False):
        if a == 0:
            if coordinates == "xyz":
                if verbose: print("Running Schwarzschild integrator in XYZ coords")
                self.gi = GeodesicIntegratorSchwarzschildXYZ(mass=mass, time_like=time_like, auto_grad = auto_grad, verbose = verbose)
            elif coordinates == "SPH2PATCH":
                if verbose: print("Running Schwarzschild integrator in SPH1SPH2 coords")
                self.gi = GeodesicIntegratorSchwarzschildSPH1SPH2(mass=mass, theta_switch = theta_switch, time_like=time_like, verbose = verbose)
            elif coordinates == "SPH": 
                self.gi = GeodesicIntegratorSchwarzschildSPH1SPH2(mass=mass, theta_switch = theta_switch, time_like=time_like, verbose = verbose)
            else:
                print("NO INTEGRATOR SELECTED")

        else:
            if verbose: print("Running KerrSchild integrator in XYZ coords")
            self.gi = GeodesicIntegratorKerrSchildXYZ(mass=mass, a=a, time_like=time_like, verbose = verbose)


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

    def get_metric(self):
        return self.gi.metric


