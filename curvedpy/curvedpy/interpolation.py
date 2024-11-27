import numpy as np
from scipy.interpolate import splev, splprep



# splprep: 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html#scipy.interpolate.splprep

# splev: 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html#scipy.interpolate.splev
class Curve:
	def __init__(self, t, x, y, z, k=3, verbose=False):
		#self.x, self.y, self.z = x, y, z

		self.tck, u = splprep([x, y, z], u=t, k=5)

	def get(self, t):
		x, y, z = splev(t, self.tck)
		return x, y, z



	# Figuring out the t for a given x, y, z would go little like this:
    # def trajectory_hit(p, args):
    #     R = args['R']
    #     #R=30
    #     p = splev([p], tck)
    #     return np.linalg.norm(p)-(R* (1+exit_tolerance))

    # sol = root(trajectory_hit, x0 = 0.5, args={"R": ratio_obj_to_blackhole})

    # p = splev(sol.x, tck)
