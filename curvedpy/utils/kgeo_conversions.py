
import numpy as np
import os
import h5py
import pickle

def kgeoToCurvedpy(fname, width, height, output_fname="", output_dir = "."):
# f.keys()
# print(f['r'][:].shape)
# print(f['r'][0].shape)
# a = f['alpha'][:]
# b = f['beta'][:]
# r, theta, phi = f['r'][:], f['theta'][:], f['phi'][:]
# a.shape[0]
# r.shape#[:].reshape(100,100,250)

	if output_fname == "":
		output_fname = fname

	f = h5py.File(fname, "r")
	a = f['alpha'][:]
	b = f['beta'][:]
	r, theta, phi = f['r'][:], f['theta'][:], f['phi'][:]

	return kgeoToCurvedpy_wrapped(width, height, a, b, r, theta, phi, output_dir, fname)

def kgeoToCurvedpy_wrapped(width, height, kgeo_impact_a, kgeo_impact_b, r, theta, phi, directory, fname):
    x = np.sqrt(r**2) * np.sin(theta) * np.cos(phi)
    y = np.sqrt(r**2) * np.sin(theta) * np.sin(phi)
    z = np.sqrt(r**2) * np.cos(theta)

    # shape x is (nr of geo, length geo), eg (262144, 248)
    x = x[1:-1].T 
    y = y[1:-1].T
    z = z[1:-1].T

    geodesics = []
    # We walk over all the geodesics and calculate the direction for every step
    for i in range(len(x)):
        kx = x[i][1:]-x[i][:-1]
        ky = y[i][1:]-y[i][:-1]
        kz = z[i][1:]-z[i][:-1]

        kx = np.append([x[i][1]-x[i][0]], kx)
        ky = np.append([y[i][1]-y[i][0]], ky)
        kz = np.append([z[i][1]-z[i][0]], kz)
        
        k = np.column_stack([kx, ky, kz])
        k = k/np.linalg.norm(k)
        
        kx, ky, kz = np.column_stack(k)
                
        geodesics.append([[kx, ky, kz], [x[i], y[i], z[i]]])

    pixel_coordinates = []
    for iy, ix in np.ndindex((height, width)):
        pixel_coordinates.append([iy, ix, 0])
    
    ray_blackhole_hit = np.zeros([height, width, 1])

    dict_to_dump = {"info": {"height":height, "width":width}, \
                    "geodesics": geodesics, \
                    "pixel_coordinates": pixel_coordinates, \
                    "ray_blackhole_hit": ray_blackhole_hit}
    with open(os.path.join(directory, fname+'.pkl'), 'wb') as f:
        pickle.dump(dict_to_dump, f)
    
    return dict_to_dump



   # def OLDconvertKgeoToCam(width, height, kgeo_impact_a, kgeo_impact_b, r, theta, phi, directory, fname):
#     # Boyer-Lindquist equations
#     x = np.sqrt(r**2) * np.sin(theta) * np.cos(phi)
#     y = np.sqrt(r**2) * np.sin(theta) * np.sin(phi)
#     z = np.sqrt(r**2) * np.cos(theta)
#     #coordinates = np.array([x, y, z])
#     x = x[1:-1].T
#     y = y[1:-1].T
#     z = z[1:-1].T
#     print(x.shape)
    
    
# #     impact_a = kgeo_impact_a.reshape(height,width)
# #     impact_b = kgeo_impact_b.reshape(height,width)
# #     geodesics = kgeodesics.T
#     geodesics = []
#     for i in range(len(x)):
#         k = []
#         for j in range(len(x[i])):
#             if j == 0:
#                 dx = x[i][j+1]-x[i][j]
#                 dy = y[i][j+1]-y[i][j]
#                 dz = z[i][j+1]-z[i][j]
#             else:
#                 dx = x[i][j]-x[i][j-1]
#                 dy = y[i][j]-y[i][j-1]
#                 dz = z[i][j]-z[i][j-1]
#             d = np.array([dx, dy, dz])
#             d = d / np.linalg.norm(d)
#             k.append(d)
        
#         kx, ky, kz = np.column_stack(k)
                
#         dummy_k = np.zeros(len(x[i]))
#         geodesics.append([[kx, ky, kz], [x[i], y[i], z[i]]])

#     print(len(geodesics))
    
#     pixel_coordinates = []
#     for iy, ix in np.ndindex((height, width)):
#         pixel_coordinates.append([iy, ix, 0])
    
#     print(len(pixel_coordinates))
#     ray_blackhole_hit = np.zeros([height, width, 1])
#     print(ray_blackhole_hit.shape)
# # #     print(ray_blackhole_hit.shape)
# # #     print(ray_blackhole_hit[99,99,0])
    
# # #     for i in range(len(pixel_coordinates)):
# # #         iy, ix, s = pixel_coordinates[i]
# # #         print(iy, ix, s)
# # #         print(ray_blackhole_hit[iy, ix, s])
# # #     print(len(geodesics), len(pixel_coordinates), ray_blackhole_hit.shape)
    
# # # #     i_geo = 0
# # # #     i_point = 0
    
# # # #     print( kgeodesics[i_point][i_geo], kgeo_impact_a[i_geo], kgeo_impact_b[i_geo] )
# # # #     print( geodesics[i_geo][i_point], pixel_coordinates[i_geo])
#     dict_to_dump = {"info": {"height":height, "width":width}, \
#                     "geodesics": geodesics, \
#                     "pixel_coordinates": pixel_coordinates, \
#                     "ray_blackhole_hit": ray_blackhole_hit}
#     with open(os.path.join(directory, fname+'.pkl'), 'wb') as f:
#         pickle.dump(dict_to_dump, f)
    
#     return dict_to_dump