import numpy as np
import curvedpy as cp
from curvedpy.geodesics.blackhole import BlackholeGeodesicIntegrator
from curvedpy.utils.utils import getImpactParam
import random
import os
import pickle 
from time import time
from multiprocess import Process, Manager, Pool, cpu_count, current_process #Queue
from functools import partial

from scipy.spatial.transform import Rotation
 
class RelativisticCamera:
    """
    A class to calculate geodesics that can be used for a camera in for example Blender.
    
    ...

    Attributes
    ----------
    gi : BlackholeGeodesicIntegrator
        Geodesic integrator used to calculate geodesics.

    ray_blackhole_hit : np.array
        An array of shape (height,width,samples) that records if the geodesic ends inside the black hole horizon

    geodesics : list
        A list of length height*width*samples containing all geodesics. Each element is an (k_xyz, x_xyz). Here 
        k_xyz is an np.array of 3-momenta (per mass) in cartesian coordinates of the geodesic. And x_xyz is an np.array
        of 3-positions in cartesian coordinates also of the geodesic.

    pixel_coordinates : numpy.array
        An array of shape (height,width,samples). Each element contains the camera pixel coordinates and the sample 
        number (y, x, s = pixel_coordinates[i]). This links the camera coordinates and sample to the list of geodesics.
        For example y, x, s = pixel_coordinates[i] gives the pixel height and width coordinates (y and x) and those belong
        to the geodesic k_xyz, x_xyz = geodesics[i].

    results : list
        A list of length height*width*samples containing the full results of all geodesic integrations

    Methods
    -------
    run_mp(verbose=False, cores = cpu_count(), *args, **kargs)
        Calculates geodesics for all photons leaving the camera.
    
    filename_suggestion()
        Returns a string using all properties of the camera which can be used for saving.

    get_camera_vis_properties(scale=5)
        Returns camera_location, corner_points and corner_rays that can be used to plot the orientation of the camera.
        # PLOTTING EXAMPLE!
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.view_init(elev=0, azim=90, roll=0)
        # lim = 5
        # ax.set_xlabel("x")
        # ax.set_xlim(-lim,lim)
        # ax.set_ylim(-lim,lim)
        # ax.set_zlim(20,31)

        # for c in corner_rays:
        #     ax.quiver(*camera_location, *c, color="k")
        # ax.plot(*list(zip(*corner_points)))

    """


    start_R = Rotation.from_euler('x', 0, degrees=True)

    def __init__(self,  camera_location = np.array([0.0001, 0, 30]), \
                        camera_rotation_euler_props=['x', 0], \
                        resolution = [64, 64],\
                        field_of_view = [0.3, 0.3],\
                        coordinates = "SPH2PATCH",\
                        M=1.0, \
                        a = 0.0,\
                        theta_switch = 0.1*np.pi,\
                        samples = 1,\
                        sampling_seed = 43,\
                        y_lim = [], x_lim = [],\
                        max_step = 1.0,\
                        loadFromFile = "",\
                        verbose=True,\
                        verbose_integrator = False,\
                        verbose_init = True):
        """Create a camera object for calculation of geodesics starting in the camera.

            Keyword arguments:
            camera_location -- camera location (default np.array([0.0001, 0, 30]) )
            camera_rotation_euler_props -- Euler rotation properties giving the orientation of the camera (default ['x', 0])
            resolution -- (default [64, 64])
            field_of_view -- (default [0.3, 0.3])
            coordinates -- (default "SPH2PATCH")
            M -- (default 1.0)
            a -- (default 0.0)
            theta_switch -- (default 0.1*np.pi)
            samples -- (default 1)
            sampling_seed -- (default 43)
            y_lim -- (default [])
            x_lim -- (default [])
            max_step -- (default 1.0)
            verbose -- (default True)
            verbose_integrator -- (default False)
            verbose_init -- (default True)
        """

        self.verbose = verbose
        
        if os.path.isfile(loadFromFile):
            self.load(loadFromFile)
        else:

            self.M = M
            self.a = a
            self.theta_switch = theta_switch

            self.camera_location = camera_location
            self.camera_rotation_euler_props = camera_rotation_euler_props
            self.camera_rotation_euler = Rotation.from_euler(camera_rotation_euler_props[0], camera_rotation_euler_props[1], degrees=True)
            self.camera_rotation_matrix = self.camera_rotation_euler.as_matrix()

            self.field_of_view = field_of_view
            self.field_of_view_x, self.field_of_view_y = self.field_of_view

            self.height, self.width = resolution

            if len(y_lim) == 0: self.y_lim = [0, self.height]
            else: self.y_lim = y_lim
            if len(x_lim) == 0: self.x_lim = [0, self.width]
            else: self.x_lim = x_lim

            self.aspectratio = self.height/self.width
            
            self.dy = self.aspectratio/self.height  
            self.dx = 1/self.width  

            random.seed(sampling_seed)  
            self.sampling_seed = sampling_seed
            self.samples = samples
            self.N = self.samples*self.width*self.height

            self.max_step = max_step
            if coordinates == "":
                coordinates = "SPH2PATCH"
            self.coordinates = coordinates
        


        # if self.theta_switch == "":
        #     self.gi = BlackholeGeodesicIntegrator(mass=self.M, a=self.a, coordinates=self.coordinates, verbose = verbose_integrator)

        # else:
        self.gi = BlackholeGeodesicIntegrator(mass=self.M, a=self.a, theta_switch = self.theta_switch, coordinates=self.coordinates, verbose = verbose_integrator)

        self.results = None

        if verbose_init:
            print("Camera Settings: ")
            print(f"  - {self.coordinates=}")
            print(f"  - {self.M=}")
            print(f"  - {self.a=}")
            print(f"  - {self.theta_switch=}")
            print(f"  - {self.verbose=}")
            print(f"  - {verbose_init=}")
            print(f"  - {self.camera_location=}")
            print(f"  - {self.camera_rotation_euler_props=}")
            print(f"  - {self.width=}")
            print(f"  - {self.height=}")
            print(f"  - {field_of_view=}")
            print(f"  - {self.samples=}")
            print(f"  - {self.sampling_seed=}")
            print(f"  - {self.y_lim=}")
            print(f"  - {self.x_lim=}")
            print(f"  - {self.max_step=}")
            #print(f"  - {force_schwarzschild_integrator=}")
            print("--")



    def filename_suggestion(self):
        """Constructs and returns a sting containing information of the camera."""

        fn = "coordinates_"+str(self.coordinates)+"_res_"+str(self.height)+"x"+str(self.width)+\
                "_fov-x_"+str(self.field_of_view_x)+"_fov-y_"+str(self.field_of_view_y)+\
                "_sample_"+str(self.samples)+"_sampling_seed_"+str(self.sampling_seed)+\
                "_a_"+str(self.a)+"_M_"+str(self.M)+\
                "_xyz0_"+str(self.camera_location[0])+"_"+str(self.camera_location[1])+"_"+str(self.camera_location[2])+\
                f"_rot_{self.camera_rotation_euler_props[0]}_angle_{self.camera_rotation_euler_props[1]}"+\
                "_max_step_"+str(self.max_step)+"_th_switch_"+str(round(self.theta_switch,3))

        return fn

    def cam_information_dict(self):
        """Internal use"""
        
        return \
            {"version_curvedpy": cp.__version__, "M":self.M, "a":self.a, "camera_location": self.camera_location, \
            "camera_rotation_euler_props":self.camera_rotation_euler_props, \
            "field_of_view": self.field_of_view, \
            "height":self.height, "width":self.width,\
            "y_lim":self.y_lim, "x_lim":self.x_lim,\
            "samples":self.samples, "sampling_seed":self.sampling_seed, \
            "max_step":self.max_step,"coordinates":self.coordinates,\
            "theta_switch":self.theta_switch\
            }
    
    def store_info_dict(self, info):
        """Dont use this"""

        #print(info)

        if cp.__version__ != info["version_curvedpy"]:
            print("WARNING DIFFERENT CURVEDPY VERSIONS!")

        self.M = info["M"]
        self.a = info["a"]
        self.camera_location = info["camera_location"]
        self.camera_rotation_euler_props = info["camera_rotation_euler_props"]
        self.field_of_view = info["field_of_view"]
        self.height = info["height"]
        self.width = info["width"]
        self.y_lim = info["y_lim"]
        self.x_lim = info["x_lim"]
        self.samples = info["samples"]
        self.sampling_seed = info["sampling_seed"]
        self.max_step = info["max_step"]
        self.coordinates = info["coordinates"]
        if "theta_switch" in info.keys():
            self.theta_switch = info["theta_switch"]
        else:
            self.theta_switch = 0.1*np.pi

    def get_x_render(self, x):
        """Dont use this"""

        return self.field_of_view_x * (x-int(self.width/2))/self.width

    def get_y_render(self, y):
        """Dont use this"""

        return self.field_of_view_y * (y-int(self.height/2))/self.height * self.aspectratio 

    def get_ray_direction(self, y_render, x_render):
        """Dont use this"""

        #ray_direction = np.array( [ x_render, y_render, -1 ] )
        ray_direction = np.array( [ x_render + self.dx*(random.random()-0.5), y_render + self.dy*(random.random()-0.5), -1 ] )
        # The ray direction relative to the camera
        ray_direction = self.camera_rotation_matrix @ ray_direction
        # Normalize the direction ray
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        return ray_direction

    def get_start_values(self, verbose=False, verbose_lvl = 1):
        """Creates starting values for the geodesics making up the camera."""

        k0, x0, pixel_coordinates = [], [], []

        for s in range(self.samples):
            for y in range(self.height):
                if y >= self.y_lim[0] and y <= self.y_lim[1]:
                    y_render = self.get_y_render(y)#self.field_of_view_y * (y-int(self.height/2))/self.height * self.aspectratio 
                    for x  in range(self.width):
                        if x >= self.x_lim[0] and x <= self.x_lim[1]:
                            x_render = self.get_x_render(x)#self.field_of_view_x * (x-int(self.width/2))/self.width

                            # The ray direction in the -z direction:
                            # ray_direction = np.array( [ x_render + self.dx*(random.random()-0.5), y_render + self.dy*(random.random()-0.5), -1 ] )
                            # ray_direction = np.array( [ x_render, y_render, -1 ] )

                            # # The ray direction relative to the camera
                            # ray_direction = self.camera_rotation_matrix @ ray_direction
                            # # Normalize the direction ray
                            # ray_direction = ray_direction / np.linalg.norm(ray_direction)

                            k0.append(self.get_ray_direction(y_render, x_render))
                            x0.append(self.camera_location)
                            pixel_coordinates.append((y,x,s))

        return k0, x0, pixel_coordinates

    def get_camera_vis_properties(self, scale=5):
        """Returns information to plot the camera orientation."""

        corners = [(0,0), (0,self.width), (self.height, self.width), (self.height, 0)]
        corner_rays = [scale*self.get_ray_direction(self.get_y_render(y), self.get_x_render(x)) for y, x in corners]
        corner_points = [self.camera_location+c for c in [*corner_rays, corner_rays[0]]] # Add the first one to make a closed loop for plotting

        return self.camera_location, corner_points, corner_rays


    def run_mp(self, verbose=False, cores = cpu_count(), *args, **kargs):
        """Calculate all geodesics starting in the camera."""

        start_time = time()

        self.ray_blackhole_hit = np.zeros(self.width * self.height * self.samples * 1)
        self.ray_blackhole_hit.shape = self.height, self.width, self.samples

        # Geodesics have different lengths, so we use lists to save them
        self.geodesics = []

        if self.verbose: print(f"Starting run_mp on {cores=}")
        k0, x0, self.pixel_coordinates = self.get_start_values()
        if self.verbose: print(f"  We have {len(k0)} models to run at {self.height}x{self.width} with sampling {self.samples}")

        split_factor = 16
        if len(k0) < split_factor:
            split_factor = 1
        start_values = list(zip(np.array_split(k0, cores*split_factor), np.array_split(x0, cores*split_factor)))
        
        if self.verbose: print(f"Multiproc info: {cores=}, {split_factor=}, {len(start_values)}, {len(start_values[0])}")
        if self.verbose: print()

        def wrap_calc_trajectory(k0_xyz, x0_xyz, shared, mes="no mes"):
            #if self.verbose: print(f"          Starting: {current_process().name}, processing array of shape: {k0_xyz.shape}")
            res = shared['gi'].geodesic(k0_xyz = k0_xyz, x0_xyz = x0_xyz, max_step = self.max_step, full_save=False, verbose=False, *args, **kargs)
            #if self.verbose: print(f"          Done: {current_process()}")
            return res

        with Manager() as manager:
            shared = manager.dict()
            shared['gi'] = self.gi
            partial_wrap_calc_trajectory = partial(wrap_calc_trajectory, shared=shared)
            with Pool(cores) as pool:
                results_pool = pool.starmap(partial_wrap_calc_trajectory, start_values)

        self.results = [ x for xs in results_pool for x in xs]

        for i in range(len(self.pixel_coordinates)):
            y, x, s = self.pixel_coordinates[i]
            
            if isinstance(self.results[i][2], list):
                self.ray_blackhole_hit[y, x, s] = int(self.results[i][2][-1].hit_blackhole)
            else:
                self.ray_blackhole_hit[y, x, s] = int(self.results[i][2]["hit_blackhole"])
            k_xyz, x_xyz = self.results[i][0], self.results[i][1]
            self.geodesics.append([k_xyz, x_xyz])

        if self.verbose: print(f"  Running is done and we have {len(self.results)} results")
        if self.verbose: print(f"Time taken for run_mp: {time()-start_time}")


    def save(self, fname, directory):
        """Saves the camera outputs and info into a pickle file."""

        if os.path.isdir(directory):
            # with open(os.path.join(directory, fname+'.pkl'), 'wb') as f:
            #     info = self.cam_information_dict()
            #     pickle.dump({"info": info, "geodesics": self.geodesics, "pixel_coordinates": self.pixel_coordinates, \
            #         "results": self.results, "ray_blackhole_hit": self.ray_blackhole_hit}, f)

            with open(os.path.join(directory, fname+'.pkl'), 'wb') as f:
                info = self.cam_information_dict()
                pickle.dump({"info": info, "geodesics": self.geodesics, "pixel_coordinates": self.pixel_coordinates, \
                    "ray_blackhole_hit": self.ray_blackhole_hit}, f)

        else:
            print("dir not found")

    def load(self, filepath):
        """Loads camera results and information into this object."""

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filecontent = pickle.load(f)
                print(filecontent.keys())
                self.store_info_dict(filecontent["info"])
                self.geodesics = filecontent["geodesics"]
                self.ray_blackhole_hit = filecontent["ray_blackhole_hit"]
                self.pixel_coordinates = filecontent["pixel_coordinates"]
        else:

            print(f"CAM: FILE NOT FOUND {filepath}")




    def calcStats(self, select_by_impact_parameter = [0, np.inf]):
        """Debug function"""

        stats = []
        for el in self.results:
            k, x, res = el
            x0 = np.column_stack(x)[0]
            #print(x0)
            k0 = np.column_stack(k)[0]
            impact_vector_normed, impact_par = getImpactParam(x0, k0)
            if impact_par >= select_by_impact_parameter[0] and impact_par <= select_by_impact_parameter[1]:
                x_end = np.column_stack(x)[-1]
                k_end = np.column_stack(k)[-1]
                deflect = k_end.dot(k0)
                hit = int(res.hit_blackhole)
                stats.append((x0, k0, x_end, k_end, impact_vector_normed, impact_par, deflect, hit))

        x0, k0, x_end, k_end, impact_vector_normed, impact_par, deflect, hit = list(zip(*stats))
        
        return {"x0":x0, "k0":k0, "x_end":x_end, "k_end":k_end, "impact_vector_normed":impact_vector_normed, \
                "impact_par":impact_par, "deflect":deflect, "hit":hit}


# https://en.wikipedia.org/wiki/Euler_angles
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html#scipy.spatial.transform.Rotation.from_euler
#start_R = Rotation.from_euler(seq='XYZ', angles=[0, 90, 0], degrees=True)


# Or do the follwoing:
# np.array(cameuler.to_matrix())
# np.array(bpy.data.scenes['Scene'].camera.matrix_world.to_euler().to_matrix())@np.array([0,0,-1])

# The standard direction from which the rotation is measured in Blender is the 0,0,-1 direction, So downwards in the z direction


# Do the following inside blender:
# Get the Euler rotation of the camera:
# cam_euler = bpy.data.scenes['Scene'].camera.matrix_world.to_euler()
# Put this rotation in a scipy rotation:
# r = Rotation.from_euler(cam_euler.order, [cam_euler.x, cam_euler.y, cam_euler.z], degrees=False)
# Give this r as camera_rotation to this class


# For more information:
# Eurler rotations in Blender are given like this:
# cameuler = C.scene.camera.matrix_world.to_euler()
# > Euler((1.1093189716339111, -0.0, 0.8149281740188599), 'XYZ')
# (https://blender.stackexchange.com/questions/130948/blender-api-get-current-location-and-rotation-of-camera-tracking-an-object)
# cameuler.order gives the 'XYZ'
# 

# In scipy a rotation can be created using:
# r = Rotation.from_euler('zyx', [90, 45, 30], degrees=True)
