
from time import time

import geodesic_integrator
GI = geodesic_integrator.GeodesicIntegrator()


#print( GI.calc_trajectory(x0=-3, y0=0, z0=0)["hit_blackhole"] )
#print("---")
_val0 = 1., -10.0, 0., 5.0, 0., 5.0

_start = time()
k_x, x, k_y, y, k_z, z = GI.calc_trajectory(*_val0, curve_start = 0, curve_end = 1, nr_points_curve = 2)["y"]
#print(x,y,z)
_vals = k_x[-1], x[-1], k_y[-1], y[-1], k_z[-1], z[-1]
k_x, x, k_y, y, k_z, z = GI.calc_trajectory(*_vals, curve_start = 1, curve_end = 2, nr_points_curve = 2)["y"]
print(time()-_start)
print(x,y,z)
print()

_start = time()
k_x, x, k_y, y, k_z, z = GI.calc_trajectory(*_val0, curve_start = 0, curve_end = 2, nr_points_curve = 3)["y"]
print(time()-_start)
print(x,y,z)
