import numpy as np
import time
from curvedpy.geodesics.blackhole_integrators.schwarzschild.special.schwarzschild_XYZ import GeodesicIntegratorSchwarzschildXYZ
from curvedpy.geodesics.blackhole_integrators.schwarzschild.special.schwarzschild_SPH1SPH2 import GeodesicIntegratorSchwarzschildSPH1SPH2
from curvedpy.geodesics.blackhole_integrators.kerr.kerr_XYZ import GeodesicIntegratorKerrSchildXYZ
from curvedpy.geodesics.blackhole_integrators.kerr.kerr_SPH import GeodesicIntegratorKerr

# 29/05/2025: This will be de definitive SS integrator based on lpn coordinates in the th=1/2pi plane.
from curvedpy.geodesics.blackhole_integrators.schwarzschild.schwarzschild_geodesic import IntegratorSchwarzschildSPH2D



class BlackholeGeodesicIntegrator:
    """
    A class to represent different black hole geodesic integrators as one object.
    
    ...

    Attributes
    ----------
    gi : multiple
        geodesic integrator used

    Methods
    -------
    geodesic(k0_xyz, x0_xyz, *args, **kargs)
        Calculates (a) geodesic(s) and returns its/their trajectories and momentum (per mass)
    """

    # FORCE_COORDINATES_CARTESIAN = {"force_coordinates": "xyz", "Explanation": ""}
    # FORCE_COORDINATES_SPH2PATCH = {"force_coordinates": "SPH2PATCH", "Explanation": ""}
    # FORCE_COORDINATES_SPH = {"force_coordinates": "SPH", "Explanation": "", "Note": "Only for debugging!"}

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, a = 0, time_like = False, coordinates = "lpn", theta_switch = 0.1*np.pi, verbose=False):
        """
        Initialize a class to represent different black hole geodesic integrators as one object.

        Parameters
        ----------
            mass : float
                Mass of the black hole in geometrized units
            a : float
                Rotation of the black hole per mass
            time_like: bool
                True for a time-like (massive particles) geodesic and False for a massless (photon) geodesic 
            coordinates : str
                Coordinates used for integration: "xyz" or "SPH2PATCH"
            theta_switch: float
                ...
            verbose: bool
                print or not to print lots of information.

        """

        if a == 0:
            if coordinates == "lpn":
                if verbose: print("Running Schwarzschild integrator in LPN coords.")
                self.gi = IntegratorSchwarzschildSPH2D(mass=mass, time_like=time_like, verbose = verbose)
            elif coordinates == "xyz": # ONLY FOR TESTING
                if verbose: print("Running Schwarzschild integrator in XYZ coords. Only for testing.")
                self.gi = GeodesicIntegratorSchwarzschildXYZ(mass=mass, time_like=time_like, verbose = verbose)
            elif coordinates == "SPH2PATCH": # ONLY FOR TESTING
                if verbose: print("Running Schwarzschild integrator in SPH1SPH2 coords. Only for testing.")
                self.gi = GeodesicIntegratorSchwarzschildSPH1SPH2(mass=mass, theta_switch = theta_switch, time_like=time_like, verbose = verbose)
            elif coordinates == "SPH": # ONLY FOR TESTING
                if verbose: print("Running Schwarzschild integrator in SPH coords. Only for testing.")
                self.gi = GeodesicIntegratorSchwarzschildSPH1SPH2(mass=mass, theta_switch = theta_switch, time_like=time_like, verbose = verbose)
            else:
                print("NO INTEGRATOR SELECTED")

        else:
            if coordinates == "BL": # ONLY FOR TESTING
                self.gi = GeodesicIntegratorKerr(mass=mass, a=a, time_like=time_like, verbose = verbose)
            else:
                if verbose: print("Running KerrSchild integrator in XYZ coords. Only for testing!")
                self.gi = GeodesicIntegratorKerrSchildXYZ(mass=mass, a=a, time_like=time_like, verbose = verbose)

    ################################################################################################
    #
    ################################################################################################
    def geodesic(self, k0_xyz, x0_xyz, *args, **kargs):
        """ Calculate a geodesic and return the coordinates and momenta (per mass).

            We always use geometrized units. Time component of the momenta is calculated using the norm of the 4 vector, 
            which is based on the time_like setting. The time component of the initial location is set to 0.

            Keyword arguments:
            k0_xyz -- initial condition, x, y and z component of the 4-momenta (per mass), numpy array of length 3. 
            x0_xyz -- initial condition, x, y and z component of the 4-location, numpy array of length 3
        """

        return self.gi.calc_trajectory(k0_xyz, x0_xyz, *args, **kargs)


    ################################################################################################
    #
    ################################################################################################
    # def get_r_s(self):
    #     """Return the radius of the black hole horizon."""
        
    #     return self.gi.metric.get_r_s()

    def get_m(self):
        """Return the mass of the black hole."""
        return self.gi.M

    def get_metric(self):
        """Get the metric used for integrating the geodesic."""
        return self.gi.metric


