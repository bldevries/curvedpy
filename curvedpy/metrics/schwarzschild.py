import sympy as sp
import numpy as np


class SchwarzschildMetricSpherical:
    
    def __init__(self):
        self.t, self.r, self.th, self.ph, self.r_s = sp.symbols("t r \\theta \\phi r_s", real=True, positive=True)

        self.g__mu__nu = sp.Matrix([\
            [-1*(1-self.r_s/self.r), 0, 0, 0],\
            [0, 1/(1-self.r_s/self.r), 0, 0],\
            [0, 0, self.r**2, 0],\
            [0, 0, 0, self.r**2 * sp.sin(self.th)**2]\
            ])

        self.g_mu_nu = self.g__mu__nu.inv()


        self.g__mu__nu_lamb = sp.lambdify([self.t, self.r, self.th, self.ph, self.r_s], self.g__mu__nu, "numpy")
        self.g_mu_nu_lamb = sp.lambdify([self.t, self.r, self.th, self.ph, self.r_s], self.g_mu_nu, "numpy")

    def oneform(self, k4_mu, x4_mu, r_s):

        if k4_mu.shape[0] == 4:
            k4_mu = np.column_stack(k4_mu)
        if x4_mu.shape[0] == 4:
            x4_mu = np.column_stack(x4_mu)

        k4__mu = np.column_stack(np.array([self.g__mu__nu_lamb(*x4_mu[i], r_s = r_s)@k4_mu[i] for i in range(len(k4_mu))]))

        return k4__mu