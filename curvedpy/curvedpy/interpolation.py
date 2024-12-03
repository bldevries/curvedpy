import numpy as np
from scipy.interpolate import splev, splprep



# splprep: 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html#scipy.interpolate.splprep
# splev: 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html#scipy.interpolate.splev
class Curve:
	def __init__(self, t, x, y, z, k=3, verbose=False):
		self.tck, u = splprep([x, y, z], u=t, k=5)

	def get(self, t):
		x, y, z = splev(t, self.tck)
		return x, y, z
