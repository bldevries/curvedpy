import sympy as sp
import numpy as np

class Conversions4D:
    def __init__(self, verbose=False):

        # NOTE TO SELF: I treat things as 3 vectors here. What if the metric has a mixed term, should I not include t? (26/11/2024)
        self.setup_sph_to_xyz_conversion()
        self.setup_xyz_to_sph_conversion()
        #self.setup_bl_to_xyz_conversion()
        #self.setup_xyz_to_bl_conversion()

    def setup_xyz_to_sph_conversion(self):
        t, x, y, z = sp.symbols("t x y z ")
        k_t, k_x, k_y, k_z = sp.symbols("k_t k_x k_y k_z")

        r = sp.sqrt(x**2+y**2+z**2)
        th = sp.acos(z/r)
        phi = sp.atan(y/x)

        M_xyz_to_sph = sp.Matrix([[1, 0, 0, 0],\
                                  [0, r.diff(x), r.diff(y), r.diff(z)],\
                                  [0, th.diff(x), th.diff(y), th.diff(z)],\
                                  [0, phi.diff(x), phi.diff(y), phi.diff(z)],\
                                 ])

        k = sp.Matrix([k_t, k_x, k_y, k_z])
        k_sph = M_xyz_to_sph*k
        self.convert_k_xyz_to_k_sph = sp.lambdify([t, x, y, z, k_t, k_x, k_y, k_z], \
                                     k_sph, "numpy")

    def setup_sph_to_xyz_conversion(self):
        t, r, th, ph = sp.symbols("t r th ph ")
        k_t, k_r, k_th, k_ph = sp.symbols("k_t k_r k_th k_ph")

        x = r * sp.sin(th) * sp.cos(ph)
        y = r * sp.sin(th) * sp.sin(ph)
        z = r * sp.cos(th)

        M_sph_to_xyz = sp.Matrix([[1, 0, 0, 0],\
                                  [0, x.diff(r), x.diff(th), x.diff(ph)],\
                                  [0, y.diff(r), y.diff(th), y.diff(ph)],\
                                  [0, z.diff(r), z.diff(th), z.diff(ph)],\
                                 ])

        v = sp.Matrix([t, r, th , ph])
        k = sp.Matrix([k_t, k_r, k_th, k_ph])
        k_xyz = M_sph_to_xyz*k
        self.convert_k_sph_to_k_xyz = sp.lambdify([t, r, th, ph, k_t, k_r, k_th, k_ph], \
                                     k_xyz, "numpy")

    ##################################################################
    # SPHERICAL - XYZ
    ##################################################################
    def convert_sph4_to_xyz4(self, k_sph, x_sph):

        # t, r, th, ph = x_sph

        # if th < 0.0: 
        #     th = abs(th)
        #     ph += np.pi

        #     x_sph = np.array([t, r, th, ph])
        #     print("DOING", x_sph)

        x_xyz = self.coord_conversion_sph4_to_xyz4(*x_sph)
        k_xyz = self.convert_k_sph_to_k_xyz(*x_sph, *k_sph)
        k_xyz = k_xyz.reshape(*k_sph.shape)
        return k_xyz, x_xyz

    def coord_conversion_sph4_to_xyz4(self, t, r, th, ph):
        z = r*np.cos(th)
        x = r*np.sin(th)*np.cos(ph)
        y = r*np.sin(th)*np.sin(ph)
        return np.array([t, x, y, z])

    ##################################################################
    # XYZ - SPHERICAL
    ##################################################################
    def convert_xyz4_to_sph4(self, k_xyz, x_xyz):
        v_sph = self.coord_conversion_xyz4_to_sph4(*x_xyz)
        k_sph = self.convert_k_xyz_to_k_sph(*x_xyz, *k_xyz)
        k_sph = k_sph.reshape(*k_xyz.shape)
        return k_sph, v_sph

    def coord_conversion_xyz4_to_sph4(self, t, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        th = np.acos(z/r)
        ph = np.atan2(y, x) #ph = np.atan(y/x)
        return np.array([t, r, th, ph])




