import numpy as np
#from curvedpy import Conversions
from curvedpy import GeodesicIntegratorKerr
import random

#import mathutils # I do not want to use this in the end but need to check with how Blender rotates things compared to scipy
from scipy.spatial.transform import Rotation
 
class RelativisticCamera:

    # https://en.wikipedia.org/wiki/Euler_angles
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html#scipy.spatial.transform.Rotation.from_euler
    #start_R = Rotation.from_euler(seq='XYZ', angles=[0, 90, 0], degrees=True)
    start_R = Rotation.from_euler('x', 0, degrees=True)

    def __init__(self,  camera_location = np.array([0.0001, 0, 10]), \
                        camera_rotation_euler=start_R, \
                        resolution = [64, 64],\
                        field_of_view = [0.3, 0.3],\
                        R_schwarzschild=1.0, \
                        a = 0.0,\
                        samples = 1,\
                        sampling_seed = 43,\
                        verbose=False):

        self.verbose = verbose

        self.M = 1/2*R_schwarzschild
        self.a = a

        self.camera_location = camera_location
        self.camera_rotation_matrix = camera_rotation_euler.as_matrix()

        self.field_of_view_x, self.field_of_view_y = field_of_view

        self.width, self.height = resolution
        self.aspectratio = self.height/self.width
        
        self.dy = self.aspectratio/self.height  
        self.dx = 1/self.width  
        random.seed(sampling_seed)  
        self.samples = samples
        self.N = self.samples*self.width*self.height
        
        self.gi = GeodesicIntegratorKerr(verbose=self.verbose, mass = self.M, a = self.a)



    def run(self):
        camera_locations = []
        ray_directions = []

        for s in range(self.samples):
            for y in range(self.height):
                y_render = self.field_of_view_y * (y-int(self.height/2))/self.height * self.aspectratio 
                for x  in range(self.width):
                    camera_locations.append(self.camera_location)

                    x_render = self.field_of_view_x * (x-int(self.width/2))/self.width

                    # The ray direction in the -z direction:
                    # ray_direction = np.array( [ x_render + self.dx*(random.random()-0.5), y_render + self.dy*(random.random()-0.5), -1 ] )
                    ray_direction = np.array( [ x_render, y_render, -1 ] )

                    # The ray direction relative to the camera
                    ray_direction = self.camera_rotation_matrix @ ray_direction
                    # Normalize the direction ray
                    ray_direction = ray_direction / np.linalg.norm(ray_direction)

                    ray_directions.append(ray_direction)


        results = self.gi.calc_trajectory( k0_xyz = ray_directions, \
                                                        x0_xyz = camera_locations,\
                                                        R_end = -1,\
                                                        curve_start = 0, \
                                                        curve_end = 50, \
                                                        nr_points_curve = 50, \
                                                        method = "RK45",\
                                                        max_step = np.inf,\
                                                        first_step = None,\
                                                        rtol = 1e-3,\
                                                        atol = 1e-6,\
                                                        verbose = self.verbose \
                                                        )


        #if result['start_inside_hole'] == False:
        #    print()

        return results


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
