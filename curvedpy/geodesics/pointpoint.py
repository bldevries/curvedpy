import numpy as np
from curvedpy.geodesics.blackhole_integrators.schwarzschild.schwarzschild_geodesic import IntegratorSchwarzschildSPH2D
from curvedpy.utils.conversions import Conversions
from time import time
conversions = Conversions()




class BlackholeGeodesicPointPointIntegrator:
    
    def __init__(self, mass=1.0, verbose=False):#coordinates = "lpn", 
        self.mass = mass
        self.time_like=True
        self.verbose = verbose


        self.gi = IntegratorSchwarzschildSPH2D(mass=self.mass, time_like=self.time_like, verbose = self.verbose)



    def geodesic_pp(self, x_f_xyz=np.array([-10,10,0]), x0_xyz = np.array([10,0,0]), image_nr=1, \
                    eps_r = 1, eps_phi=0.00000000005):

        if np.all(x_f_xyz == x0_xyz):
            print(f"Start and end point are the same: {x_f_xyz=} {x0_xyz=}")

        y_is_neg = False
        if x_f_xyz[1] < 0:
            y_is_neg = True
            x_f_xyz[1] = -x_f_xyz[1]
            
        
        mass = 1.0
        time_like = True
        verbose = False
        fmt = ["-x", "--", "-."]
        c = ["k", "b", "g", "r", "y"]

        if image_nr == 2:
            secondary_image = True
            third_image = False#True
        elif image_nr == 3:
            secondary_image = False
            third_image = True
        else: #if image_nr == 1:
            secondary_image = False
            third_image = False#True

        if secondary_image: # secondary_image
            # print(image_nr, not third_image, secondary_image, third_image)
            x_f_xyz_use = np.array([x_f_xyz[0], -x_f_xyz[1], x_f_xyz[2]])
        else:
            x_f_xyz_use = np.copy(x_f_xyz)
            
        r_f, _, phi_f = conversions.coord_conversion_xyz_to_sph(*x_f_xyz_use)
        print(r_f, phi_f, phi_f+2*np.pi)
        if third_image or secondary_image: # secondary_image
            phi_f = phi_f + 2*np.pi
         

        # if False:#first_image:
        phi_high = 4/4*np.pi
        phi_low = 0/2*np.pi
        # else:
        #     phi_low = -4/4*np.pi
        #     phi_high = 0/2*np.pi

        phi_mean = (phi_low+phi_high)/2
        r_high = 200
        r_low = 0

        counter=0
        while abs(r_high-r_low)>=eps_r and abs(phi_high-phi_low)>=eps_phi:# and counter<1:
            l_r_end = []
            l_k, l_x, l_results = [], [], []
            for i, phi in enumerate([phi_low, phi_mean, phi_high]):
                k0_xyz = np.array([np.cos(phi), np.sin(phi), 0])
                k0_xyz = k0_xyz/np.linalg.norm(k0_xyz)
                x0_sph, k0_sph = conversions.convert_xyz_to_sph(x0_xyz, k0_xyz)
                # print(phi, x0_xyz, k0_xyz)
                k, x, r = self.gi.calc_trajectory_sph(k0_sph, x0_sph, \
                            phi_end=phi_f,\
                            R_end = -1,\
                            curve_start = 0, curve_end = 1000, nr_points_curve = 10000, \
                            max_step = 1, first_step = None)
                # print(r)
                if len(r['y_events'][1]) > 0:
                    phi_end, r_end = r['y_events'][1][0][-1], r['y_events'][1][0][-3]
                else:
                    r_end, _, phi_end = x.T[-1]

                l_r_end.append(r_end)
                x_end, y_end, z_end = conversions.coord_conversion_sph_to_xyz(r_end, 1/2*np.pi, phi_end)
                # print(phi_f, phi_end,  abs(phi_high-phi_low), r_f, r_end, abs(r_high-r_low))
                # plt.plot(x_end, y_end, "o")
               
                x_xyz, k_xyz = conversions.convert_sph_to_xyz(x, k)
                if i == 1:
                    l_k.append(k_xyz)
                    if secondary_image:
                        x_xyz[1] = -x_xyz[1]
                    if y_is_neg:
                        x_xyz[1] = -x_xyz[1]
                        x_f_xyz[1] = -x_f_xyz[1]
                    # if third_image:
                    #     x_xyz[1] = -x_xyz[1]
                    #     x_f_xyz[1] = -x_f_xyz[1]
                        
                        
                    l_x.append(x_xyz)
                    l_results.append(r)
                # plt.plot(*x0_xyz[0:2], "o")
                # if i == 1:
                    # plt.plot(*x_xyz[0:2])#, c[counter]+fmt[i])

            if r_f > l_r_end[0]: # r_f > r(phi_low)
                phi_high = phi_low
                phi_low = phi_low - 0.1*np.pi
            elif r_f < l_r_end[2]:
                phi_low = phi_high
                phi_high = phi_high +0.1*np.pi
            elif r_f > l_r_end[1]:
                phi_high = phi_mean
                # phi_lo
            elif r_f < l_r_end[1]:
                phi_low = phi_mean

            phi_mean = (phi_low+phi_high)/2
            r_high = l_r_end[0]
            r_low = l_r_end[2]
            
            # print(counter, abs(r_high-r_low)>=eps_r, abs(phi_high-phi_low)>=eps_phi, r_high, r_low)

            counter+=1
        #plt.plot(*x_xyz[0:2])#, c[counter]+fmt[i])

        k0_xyz = l_k[-1].T[0]

        x_final = l_x[-1].T
        length = sum([np.linalg.norm(x_final[i+1]-x_final[i]) for i in range(len(x_final)-1)])
        length += np.linalg.norm(x_f_xyz-x_final[-1])

        return k0_xyz, length, l_k, l_x, l_results



