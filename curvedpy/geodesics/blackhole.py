import numpy as np
import time
from curvedpy.geodesics.blackhole_integrators.kerr.kerr_BL_geodesic import GeodesicIntegratorKerr
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

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, a = 0, time_like = False, verbose=False):#coordinates = "lpn", 
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
            verbose: bool
                print or not to print lots of information.

        """

        if a == 0:
            self.gi = IntegratorSchwarzschildSPH2D(mass=mass, time_like=time_like, verbose = verbose)
        else:
            self.gi = GeodesicIntegratorKerr(mass=mass, a=a, time_like=time_like, verbose = verbose)

    ################################################################################################
    #
    ################################################################################################
    def geodesic(self,  k0_xyz, x0_xyz, R_end = -1,\
                        curve_start = 0, curve_end = 50, nr_points_curve = 50, \
                        max_step = np.inf, first_step = None):
        """ Calculate a geodesic and return the coordinates and momenta (per mass).

            We always use geometrized units. Time component of the momenta is calculated using the norm of the 4 vector, 
            which is based on the time_like setting. The time component of the initial location is set to 0.

            Keyword arguments:
            k0_xyz -- initial condition, x, y and z component of the 4-momenta (per mass), numpy array of length 3. 
            x0_xyz -- initial condition, x, y and z component of the 4-location, numpy array of length 3
        """

        return self.gi.calc_trajectory(k0_xyz, x0_xyz, R_end,\
                        curve_start, curve_end, nr_points_curve, \
                        max_step, first_step)


    ################################################################################################
    #
    ################################################################################################
    def get_m(self):
        """Return the mass of the black hole."""
        return self.gi.M

    def get_a(self):
        return self.gi.a_value

    def get_metric(self):
        """Get the metric used for integrating the geodesic."""
        return self.gi.metric


