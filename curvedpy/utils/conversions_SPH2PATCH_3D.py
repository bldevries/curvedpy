import sympy as sp
import numpy as np


#################################################################################
# TO CART
#################################################################################
def make_J_sph1_to_cart():
    r, th, ph = sp.symbols("r \\theta \\phi")
    x = r * sp.sin(th)*sp.cos(ph)
    y = r * sp.sin(th)*sp.sin(ph)
    z = r * sp.cos(th)
    
    matrix_J_r_th_ph = sp.simplify(\
                            sp.Matrix([[x.diff(r), x.diff(th), x.diff(ph)], \
                           [y.diff(r), y.diff(th), y.diff(ph)], \
                           [z.diff(r), z.diff(th), z.diff(ph)]]))
   
    J = sp.lambdify([r, th, ph], matrix_J_r_th_ph, 'numpy')
    C = sp.lambdify([r, th, ph], [x, y, z], 'numpy')
    Det = sp.lambdify([r, th, ph], matrix_J_r_th_ph.det(), 'numpy')

    return J, C, Det, 

J_sph1_to_cart, coords_sph1_to_cart, Det_sph1_to_cart = make_J_sph1_to_cart()


def make_J_sph2_to_cart():
    r, th2, ph2 = sp.symbols("r \\theta2 \\phi2")
    x = r * sp.sin(th2) * sp.cos(ph2)
    y = r * sp.cos(th2)
    z = r * sp.sin(th2) * sp.sin(ph2)

    matrix_J_r_th2_ph2 = sp.simplify(
                            sp.Matrix([[x.diff(r), x.diff(th2), x.diff(ph2)], \
                           [y.diff(r), y.diff(th2), y.diff(ph2)], \
                           [z.diff(r), z.diff(th2), z.diff(ph2)]]))
   
    J = sp.lambdify([r, th2, ph2], matrix_J_r_th2_ph2, 'numpy')
    C = sp.lambdify([r, th2, ph2], [x, y, z], 'numpy')
    Det = sp.lambdify([r, th2, ph2], matrix_J_r_th2_ph2.det(), 'numpy')

    return J, C, Det, 

J_sph2_to_cart, coords_sph2_to_cart, Det_sph2_to_cart = make_J_sph2_to_cart()



#################################################################################
# FROM CART
#################################################################################
def make_J_cart_to_sph2():
    x, y, z = sp.symbols("x y z")
    r = sp.sqrt(x**2+y**2+z**2)
    th2 = sp.acos(y/r)
    ph2 = sp.atan2(z,x)
    
    matrix_J_x_y_z = sp.simplify(sp.Matrix([[r.diff(x), r.diff(y), r.diff(z)], \
                               [th2.diff(x), th2.diff(y), th2.diff(z)],\
                               [ph2.diff(x), ph2.diff(y), ph2.diff(z)]]))
    
    J = sp.lambdify([x, y, z], matrix_J_x_y_z, 'numpy')
    C = sp.lambdify([x, y, z], [r, th2, ph2], 'numpy')
    Det = sp.lambdify([x, y, z], matrix_J_x_y_z.det(), 'numpy')

    return J, C, Det

J_cart_to_sph2, coords_cart_to_sph2, Det_cart_to_sph2 = make_J_cart_to_sph2()


def make_J_cart_to_sph1():
    x, y, z = sp.symbols("x y z")
    r = sp.sqrt(x**2+y**2+z**2)
    th = sp.acos(z/r)
    ph = sp.atan2(y, x)
    
    matrix_J_x_y_z = sp.simplify(sp.Matrix([[r.diff(x), r.diff(y), r.diff(z)], \
                               [th.diff(x), th.diff(y), th.diff(z)],\
                               [ph.diff(x), ph.diff(y), ph.diff(z)]]))
    
    J = sp.lambdify([x, y, z], matrix_J_x_y_z, 'numpy')
    C = sp.lambdify([x, y, z], [r, th, ph], 'numpy')
    Det = sp.lambdify([x, y, z], matrix_J_x_y_z.det(), 'numpy')

    return J, C, Det

J_cart_to_sph1, coords_cart_to_sph1, Det_cart_to_sph1 = make_J_cart_to_sph1()



#################################################################################
# Check of determinant to see if coord transformation is defined
#################################################################################

def det_check_sph1_to_sph2(x_sph1):
    x_cart = coords_sph1_to_cart(*x_sph1)
    return not ((Det_sph1_to_cart(*x_sph1) == 0.0) or (Det_cart_to_sph2(*x_cart) == 0.0))

def det_check_sph2_to_sph1(x_sph2):
    x_cart = coords_sph2_to_cart(*x_sph2)
    return not ((Det_sph2_to_cart(*x_sph2) == 0.0) or (Det_cart_to_sph1(*x_cart) == 0.0))




#################################################################################
# BETWEEN SPH1 and SPH2 patches
#################################################################################

def sph1_to_cart(k, x):
    if len(k.shape) == 1:
        return J_sph1_to_cart(*x)@k, coords_sph1_to_cart(*x)
    else:
        k, x = np.column_stack(k), np.column_stack(x)
        k_xyz = np.column_stack(np.array([J_sph1_to_cart(*x[i])@k[i] for i in range(len(x))]))
        x_xyz = np.column_stack(np.array([coords_sph1_to_cart(*x[i]) for i in range(len(x))]))
        return k_xyz, x_xyz

def cart_to_sph2(k, x):
    if len(k.shape) == 1:
        return J_cart_to_sph2(*x)@k, coords_cart_to_sph2(*x)
    else:
        k, x = np.column_stack(k), np.column_stack(x)
        k_sph2 = np.column_stack(np.array([J_cart_to_sph2(*x[i])@k[i] for i in range(len(x))]))
        x_sph2 = np.column_stack(np.array([coords_cart_to_sph2(*x[i]) for i in range(len(x))]))
        return k_sph2, x_sph2

def sph1_to_sph2(k, x):
    k_cart, x_cart = sph1_to_cart(k, x)
    k_sph2, x_sph2 = cart_to_sph2(k_cart, x_cart)
    return k_sph2, x_sph2


def sph2_to_cart(k, x):
    if len(k.shape) == 1:
        return J_sph2_to_cart(*x)@k, coords_sph2_to_cart(*x)
    else:
        k, x = np.column_stack(k), np.column_stack(x)
        k_xyz = np.column_stack(np.array([J_sph2_to_cart(*x[i])@k[i] for i in range(len(x))]))
        x_xyz = np.column_stack(np.array([coords_sph2_to_cart(*x[i]) for i in range(len(x))]))
        return k_xyz, x_xyz


def cart_to_sph1(k, x):
    if len(k.shape) == 1:
        return J_cart_to_sph1(*x)@k, coords_cart_to_sph1(*x)
    else:
        k, x = np.column_stack(k), np.column_stack(x)
        k_sph1 = np.column_stack(np.array([J_cart_to_sph1(*x[i])@k[i] for i in range(len(x))]))
        x_sph1 = np.column_stack(np.array([coords_cart_to_sph1(*x[i]) for i in range(len(x))]))
        return k_sph1, x_sph1

def sph2_to_sph1(k, x):
    k_cart, x_cart = sph2_to_cart(k, x)
    k_sph1, x_sph1 = cart_to_sph1(k_cart, x_cart)
    return k_sph1, x_sph1


