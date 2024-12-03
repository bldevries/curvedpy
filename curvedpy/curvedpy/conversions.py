import sympy as sp
import numpy as np

class Conversions:
    def __init__(self, verbose=False):

        # NOTE TO SELF: I treat things as 3 vectors here. What if the metric has a mixed term, should I not include t? (26/11/2024)
        self.setup_sph_to_xyz_conversion()
        self.setup_xyz_to_sph_conversion()



    def setup_xyz_to_sph_conversion(self):
        x, y, z = sp.symbols(" x y z ")
        k_x, k_y, k_z = sp.symbols(" k_x k_y k_z")

        r = sp.sqrt(x**2+y**2+z**2)
        th = sp.acos(z/r)
        phi = sp.atan(y/x)

        M_xyz_to_sph = sp.Matrix([[r.diff(x), r.diff(y), r.diff(z)],\
                                  [th.diff(x), th.diff(y), th.diff(z)],\
                                  [phi.diff(x), phi.diff(y), phi.diff(z)],\
                                 ])

        k = sp.Matrix([k_x, k_y, k_z])
        k_sph = M_xyz_to_sph*k
        self.convert_k_xyz_to_k_sph = sp.lambdify([x, y, z, k_x, k_y, k_z], \
                                     k_sph, "numpy")

    def setup_sph_to_xyz_conversion(self):
        r, th, ph = sp.symbols(" r th ph ")
        k_r, k_th, k_ph = sp.symbols(" k_r k_th k_ph")

        x = r * sp.sin(th) * sp.cos(ph)
        y = r * sp.sin(th) * sp.sin(ph)
        z = r * sp.cos(th)

        M_sph_to_xyz = sp.Matrix([[x.diff(r), x.diff(th), x.diff(ph)],\
                                  [y.diff(r), y.diff(th), y.diff(ph)],\
                                  [z.diff(r), z.diff(th), z.diff(ph)],\
                                 ])

        v = sp.Matrix([r, th , ph])
        k = sp.Matrix([k_r, k_th, k_ph])
        k_xyz = M_sph_to_xyz*k
        self.convert_k_sph_to_k_xyz = sp.lambdify([r, th, ph, k_r, k_th, k_ph], \
                                     k_xyz, "numpy")

    def convert_sph_to_xyz(self, x_sph, k_sph):
        v_xyz = self.coord_conversion_sph_to_xyz(*x_sph)
        k_xyz = self.convert_k_sph_to_k_xyz(*x_sph, *k_sph)
        k_xyz = k_xyz.reshape(*k_sph.shape)
        return v_xyz, k_xyz
    def convert_xyz_to_sph(self, x_xyz, k_xyz):
        v_sph = self.coord_conversion_xyz_to_sph(*x_xyz)
        k_sph = self.convert_k_xyz_to_k_sph(*x_xyz, *k_xyz)
        k_sph = k_sph.reshape(*k_xyz.shape)
        return v_sph, k_sph

    def coord_conversion_sph_to_xyz(self, r, th, ph):
        z = r*np.cos(th)
        x = r*np.sin(th)*np.cos(ph)
        y = r*np.sin(th)*np.sin(ph)
        return np.array([x, y, z])

    def coord_conversion_xyz_to_sph(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        th = np.acos(z/r)
        ph = np.atan2(y, x) #ph = np.atan(y/x)
        return r, th, ph