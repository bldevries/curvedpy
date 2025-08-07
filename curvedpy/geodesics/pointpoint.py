import numpy as np
from curvedpy.geodesics.blackhole_integrators.schwarzschild.schwarzschild_geodesic import IntegratorSchwarzschildSPH2D
from curvedpy.utils.conversions import Conversions
from time import time
conversions = Conversions()




class BlackholeGeodesicPointPointIntegrator:
    """
    A class to calculate Schwarzschild geodesics between two points. Up to three geodesics are calculated 
    that connect two points in space-time. 
    
    ...

    Attributes
    ----------
    gi : the geodesic integrator used

    Methods
    -------
    geodesic_pp(self, x_f_xyz=np.array([[-10,10,0]]), x0_xyz = np.array([[10,0,0]]), image_nr=1, \
                    max_step = 1.0, eps_r = 0.0001, eps_phi=0.00000000005, max_iterations = 100, verbose=False, \
                    return_matrices = False)

        Calculates (a) geodesic(s) between the point x_f_xyz and x0_xyz for the order image_nr. It returns its/their 
        trajectories (l_x_xyz) as well as the displacement of x_f_xyz from when the trajectory would have been a 
        straight line (l_translation_xyz).
    """
    
    ##################################################################################################
    def __init__(self, mass=1.0, verbose=False):#coordinates = "lpn", 
        """
        Initialize a BlackholeGeodesicPointPointIntegrator class

        Parameters
        ----------
            mass : float
                Mass of the black hole in geometrized units
            verbose: bool
                print or not to print lots of information.

        """

        time_like=True
        self.gi = IntegratorSchwarzschildSPH2D(mass=mass, time_like=time_like, verbose = verbose)

    ##################################################################################################
    def unit_vectors_lpn(self, x0, x):
        ''' 
        Gives the unit vectors (l, p, n) of the plane at theta=1/2pi with the black hole at xyz = (0,0,0).
        p is the impact vector direction, l is the distance direction from the BH.
        x0: 3-vector, location of the start point of a geodesic, measured from the BH at xyz=(0,0,0)
        x: 3-vector, end location of the geodesic, measured from the BH
        Both need to have shape (a, 3) where a != 0
        NOTE: because the p_hat is in the direction of x, the end point will always lie in the
        positive p half of the space!
        '''

        l_ = x0
        n_ = np.cross(l_, x)
        p_ = np.cross(n_, l_)

        norm_l = np.linalg.norm(l_, axis=1)
        norm_n = np.linalg.norm(n_, axis=1)
        norm_p = np.linalg.norm(p_, axis=1)

        l_hat = (1./norm_l * l_.T).T
        p_hat = (1./norm_p * p_.T).T
        n_hat = (1./norm_n * n_.T).T

        return l_hat, p_hat, n_hat

    ##################################################################################################
    def matrix_conversion_lpn_xyz(self, l_hat, p_hat, n_hat):
        """ Matrices to go from xyz to lpn coordinate system and back"""

        # Can this be faster without list comprehension!!

        M_lpn_xyz = [np.array([l_hat[i], p_hat[i], n_hat[i]]).T for i in range(l_hat.shape[0])]
        # It might be this: 
        #M_lpn_xyz = np.einsum("ijk->jik", np.array([l_hat, p_hat, n_hat]))

        #M_xyz_lpn = [np.linalg.inv(M) for M in M_lpn_xyz]
        M_xyz_lpn = np.linalg.inv(M_lpn_xyz)

        return M_xyz_lpn, M_lpn_xyz#np.linalg.inv(M_xyz_lpn)

    ##################################################################################################
    def geodesic_pp_projection_matrix(self, x_f_xyz=np.array([-10,10,0]), x0_xyz = np.array([10,0,0]), image_nr=1, \
                    max_step = 1.0, eps_r = 0.0001, eps_phi=0.00000000005, max_iterations = 100, verbose=False):
        # NO VECTORIZATION YET!!
        l_x_xyz, l_translation_xyz = self.geodesic_pp(\
            x_f_xyz=np.array([x_f_xyz]), x0_xyz = np.array([x0_xyz]), image_nr=image_nr, \
            max_step = max_step, eps_r = eps_r, eps_phi=eps_phi, max_iterations = max_iterations, \
            verbose=verbose)

        M_tr = [[1,0,0, l_translation_xyz[0][0]],\
                [0,1,0, l_translation_xyz[0][1]],\
                [0,0,1, l_translation_xyz[0][2]],\
                [0,0,0,1]]

        return M_tr


    ##################################################################################################
    def geodesic_pp(self, x_f_xyz=np.array([[-10,10,0]]), x0_xyz = np.array([[10,0,0]]), image_nr=1, \
                    max_step = 1.0, eps_r = 0.0001, eps_phi=0.00000000005, max_iterations = 100, verbose=False, \
                    return_matrices = False):
        
        M_xyz_lpn, M_lpn_xyz = self.matrix_conversion_lpn_xyz(*self.unit_vectors_lpn(x0_xyz, x_f_xyz))

        x0_lpn = np.einsum('bij,bj->bi', M_xyz_lpn, x0_xyz)
        x_f_lpn = np.einsum('bij,bj->bi', M_xyz_lpn, x_f_xyz)

        # THIS NEEDS VECTORIZATION!
        l_x_lpn = []
        l_x_xyz = []
        l_translation_xyz = []
        for i in range(len(x0_lpn)):
            translation_lpn, _, _, _, x_lpn, _ = self.geodesic_pp_lpn_no_vec(x_f_lpn[i], x0_lpn[i], image_nr=image_nr, \
                    max_step = max_step, eps_r = eps_r, eps_phi=eps_phi, max_iterations = max_iterations, verbose=verbose)
            l_x_lpn.append(np.array(x_lpn).T)
            l_x_xyz.append(M_lpn_xyz[i]@np.array(x_lpn))

            translation_xyz = M_lpn_xyz[i] @ translation_lpn
            l_translation_xyz.append(translation_xyz)

        if return_matrices:
            return l_x_xyz, np.array(l_translation_xyz), M_xyz_lpn, M_lpn_xyz
        else:
            return l_x_xyz, np.array(l_translation_xyz)


    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    def geodesic_pp_lpn_no_vec(self, x_f_lpn=np.array([-10,10,0]), x0_lpn = np.array([10,0,0]), image_nr=1, \
                    max_step = 1.0, eps_r = 0.0001, eps_phi=0.00000000005, max_iterations = 100, debug=False, verbose=False):

        debug_log = []

        if np.all(x_f_lpn == x0_lpn):
            print(f"Start and end point are the same: {x_f_lpn=} {x0_lpn=}")

        if x_f_lpn[1] < 0:
            print(f"WARNING! If you converted properly to the lpn space, your p coordinate can not be negative {x_f_lpn=}")
            return 

        if np.linalg.norm(x_f_lpn) <= 2*self.gi.M:
            print(f"WARNING! The end point is inside the event horizon! {x_f_lpn=}")

        if np.linalg.norm(x0_lpn) <= 2*self.gi.M:
            print(f"WARNING! The start point is inside the event horizon! {x0_lpn=}")


        # The integrator only goes towards increasing phi, so this determines that we need to go from the point of 
        # small phi to that on large phi
        r_0, _, phi_0 = conversions.coord_conversion_xyz_to_sph(*x0_lpn)
        r_f, _, phi_f = conversions.coord_conversion_xyz_to_sph(*x_f_lpn)
        if phi_0 > phi_f:
            flip_positions = True
            _ = x0_lpn
            x0_lpn = x_f_lpn
            x_f_lpn = _
        else:
            flip_positions = False
        
        # We can do 3 images in total
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
            # If we want the second image, we flip the p coordinate, but still we integrate
            # in the positive phi direction. Flipping the geodesic at the end gives you
            # you the secondary image.
            # I can not just increase phi_f with np.pi, like I do for the third image 
            # (phi_f = phi_f + 2*np.pi) since the second image geodesic really goes into
            # the negative phi direction.
            x_f_lpn_use = np.copy(np.array([x_f_lpn[0], -x_f_lpn[1], x_f_lpn[2]]))
        else:
            x_f_lpn_use = np.copy(x_f_lpn)


        r_0, _, phi_0 = conversions.coord_conversion_xyz_to_sph(*x0_lpn)
        r_f, _, phi_f = conversions.coord_conversion_xyz_to_sph(*x_f_lpn_use)

        if verbose: print(f"Running pp integration: {flip_positions=}, {x0_lpn=}, {x_f_lpn_use=}")

        # To get the third image we force the integrator to integrate to phi_f + 2*np.pi, thus
        # the geodesic makes another orbit around the BH but still goes around the same direction
        # as the first image
        if third_image or secondary_image: # secondary_image
            phi_f = phi_f + 2*np.pi
         

        # Our first aim at the end point is very broad.
        # This could be optimized to be a bit smarter
        phi_high = 4/4*np.pi
        phi_low = 0/2*np.pi
        # The mean of the phi is our intermediate aim        
        phi_mean = (phi_low+phi_high)/2
        # For the first iteration we set the r_high and low 
        # Later these are set to the r value of the phi_high and low
        # geodesics to check convergence
        r_high = 200
        r_low = 0

        counter=0
        while abs(r_high-r_low)>=eps_r and abs(phi_high-phi_low)>=eps_phi and counter<max_iterations:
            l_r_end = []
            l_k, l_x, l_results = [], [], []
            log = {}
            for i, phi in enumerate([phi_low, phi_mean, phi_high]):
                k0_lpn = np.array([np.cos(phi), np.sin(phi), 0])
                k0_lpn = k0_lpn/np.linalg.norm(k0_lpn)
                x0_sph, k0_sph = conversions.convert_xyz_to_sph(x0_lpn, k0_lpn)
                k, x, r = self.gi.calc_trajectory_sph(k0_sph, x0_sph, \
                            phi_end=phi_f,\
                            R_end = -1,\
                            curve_start = 0, curve_end = 1000, nr_points_curve = 10000, \
                            max_step = max_step, first_step = None)
                # print(r['y_events'])
                if len(r['y_events'][1]) > 0:
                    k_t_end, k_r_end, k_ph_end, t_end, r_end, phi_end = r['y_events'][1][0]
                    # phi_end, r_end = r['y_events'][1][0][-1], r['y_events'][1][0][-3]
                else:
                    r_end, _, phi_end = x.T[-1]


                # print(phi, x0_xyz, k0_xyz, r_end, len(r['y_events'][1]) > 0)

                l_r_end.append(r_end)
                x_end, y_end, z_end = conversions.coord_conversion_sph_to_xyz(r_end, 1/2*np.pi, phi_end)
                # print(phi_f, phi_end,  abs(phi_high-phi_low), r_f, r_end, abs(r_high-r_low))
                # plt.plot(x_end, y_end, "o")
               
                x_xyz, k_xyz = conversions.convert_sph_to_xyz(x, k)

                log.update({str(i):x_xyz})

                if i == 1:
                    l_k.append(k_xyz)
                    if secondary_image:
                        x_xyz[1] = -x_xyz[1]
                        k_xyz[1] = -k_xyz[1]
                    # if y_is_neg:
                    #     x_xyz[1] = -x_xyz[1]
                    #     x_f_xyz[1] = -x_f_xyz[1]
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
                log.update({"action":1})
            elif r_f < l_r_end[2]:
                if phi_0 < phi_f:
                    phi_low = phi_high
                    phi_high = phi_high +0.1*np.pi
                elif phi_0 > phi_f:
                    phi_high = phi_low
                    phi_low = phi_low -0.1*np.pi
                log.update({"action":2})

            elif r_f > l_r_end[1]:
                phi_high = phi_mean
                # phi_lo
                log.update({"action":3})

            elif r_f < l_r_end[1]:
                phi_low = phi_mean
                log.update({"action":4})
            else:
                log.update({"action":5})



            phi_mean = (phi_low+phi_high)/2
            r_high = l_r_end[0]
            r_low = l_r_end[2]
            
            # print(counter, abs(r_high-r_low)>=eps_r, abs(phi_high-phi_low)>=eps_phi, r_high, r_low)

            log.update({"counter": counter})
            log.update({"flip": flip_positions})

            debug_log.append(log)

            counter+=1

        #plt.plot(*x_xyz[0:2])#, c[counter]+fmt[i])

        if verbose: print(f"{counter=}")

        if flip_positions:
            l_k = np.array([np.flip(i, 1) for i in l_k])
            l_x = np.array([np.flip(i, 1) for i in l_x])



        # if flip_positions:
        #     k0_xyz = l_k[-1].T[-1]
        # else:
        #     k0_xyz = l_k[-1].T[0]
        
        k0_lpn = l_k[-1].T[0]


        x_final = l_x[-1].T
        length = sum([np.linalg.norm(x_final[i+1]-x_final[i]) for i in range(len(x_final)-1)])
        length += np.linalg.norm(x_f_lpn_use-x_final[-1])

        if secondary_image:
            x_f_lpn_use[1] = -x_f_lpn_use[1]
        translation_lpn = x0_lpn + k0_lpn/np.linalg.norm(k0_lpn) * length - x_f_lpn_use#l_x[-1].T[-1]

        if debug:
            return translation_lpn, k0_lpn, length, l_k[0], l_x[0], l_results, debug_log
        else:
            return translation_lpn, k0_lpn, length, l_k[0], l_x[0], l_results


    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    # def geodesic_pp_xyz(self, x_f_xyz=np.array([-10,10,0]), x0_xyz = np.array([10,0,0]), image_nr=1, \
    #                 max_step = 1.0, eps_r = 1, eps_phi=0.00000000005, debug=False, verbose=False):

    #     debug_log = []

    #     if np.all(x_f_xyz == x0_xyz):
    #         print(f"Start and end point are the same: {x_f_xyz=} {x0_xyz=}")


    #     y_is_neg = False
    #     if x_f_xyz[1] < 0:
    #         y_is_neg = True
    #         x_f_xyz[1] = -x_f_xyz[1]

    #     r_0, _, phi_0 = conversions.coord_conversion_xyz_to_sph(*x0_xyz)
    #     r_f, _, phi_f = conversions.coord_conversion_xyz_to_sph(*x_f_xyz)

    #     if phi_0 > phi_f:
    #         flip_positions = True
    #         _ = x0_xyz
    #         x0_xyz = x_f_xyz
    #         x_f_xyz = _
    #     else:
    #         flip_positions = False
        
        
    #     mass = 1.0
    #     time_like = True
    #     verbose = False
    #     fmt = ["-x", "--", "-."]
    #     c = ["k", "b", "g", "r", "y"]

    #     if image_nr == 2:
    #         secondary_image = True
    #         third_image = False#True
    #     elif image_nr == 3:
    #         secondary_image = False
    #         third_image = True
    #     else: #if image_nr == 1:
    #         secondary_image = False
    #         third_image = False#True

    #     if secondary_image: # secondary_image
    #         # print(image_nr, not third_image, secondary_image, third_image)
    #         x_f_xyz_use = np.array([x_f_xyz[0], -x_f_xyz[1], x_f_xyz[2]])
    #     else:
    #         x_f_xyz_use = np.copy(x_f_xyz)


    #     r_0, _, phi_0 = conversions.coord_conversion_xyz_to_sph(*x0_xyz)
    #     r_f, _, phi_f = conversions.coord_conversion_xyz_to_sph(*x_f_xyz_use)

    #     if verbose: print(f"Running pp integration: {flip_positions=}, {x0_xyz=}, {x_f_xyz_use=}")

    #     # print(r_f, phi_f, phi_f+2*np.pi)
    #     if third_image or secondary_image: # secondary_image
    #         phi_f = phi_f + 2*np.pi
         

    #     # if False:#first_image:
    #     phi_high = 4/4*np.pi
    #     phi_low = 0/2*np.pi
    #     # else:
    #     #     phi_low = -4/4*np.pi
    #     #     phi_high = 0/2*np.pi

    #     phi_mean = (phi_low+phi_high)/2
    #     r_high = 200
    #     r_low = 0

    #     counter=0
    #     while abs(r_high-r_low)>=eps_r and abs(phi_high-phi_low)>=eps_phi:# and counter<1:
    #         l_r_end = []
    #         l_k, l_x, l_results = [], [], []
    #         log = {}
    #         for i, phi in enumerate([phi_low, phi_mean, phi_high]):
    #             k0_xyz = np.array([np.cos(phi), np.sin(phi), 0])
    #             k0_xyz = k0_xyz/np.linalg.norm(k0_xyz)
    #             x0_sph, k0_sph = conversions.convert_xyz_to_sph(x0_xyz, k0_xyz)
    #             k, x, r = self.gi.calc_trajectory_sph(k0_sph, x0_sph, \
    #                         phi_end=phi_f,\
    #                         R_end = -1,\
    #                         curve_start = 0, curve_end = 1000, nr_points_curve = 10000, \
    #                         max_step = max_step, first_step = None)
    #             # print(r['y_events'])
    #             if len(r['y_events'][1]) > 0:
    #                 k_t_end, k_r_end, k_ph_end, t_end, r_end, phi_end = r['y_events'][1][0]
    #                 # phi_end, r_end = r['y_events'][1][0][-1], r['y_events'][1][0][-3]
    #             else:
    #                 r_end, _, phi_end = x.T[-1]


    #             # print(phi, x0_xyz, k0_xyz, r_end, len(r['y_events'][1]) > 0)

    #             l_r_end.append(r_end)
    #             x_end, y_end, z_end = conversions.coord_conversion_sph_to_xyz(r_end, 1/2*np.pi, phi_end)
    #             # print(phi_f, phi_end,  abs(phi_high-phi_low), r_f, r_end, abs(r_high-r_low))
    #             # plt.plot(x_end, y_end, "o")
               
    #             x_xyz, k_xyz = conversions.convert_sph_to_xyz(x, k)

    #             log.update({str(i):x_xyz})

    #             if i == 1:
    #                 l_k.append(k_xyz)
    #                 if secondary_image:
    #                     x_xyz[1] = -x_xyz[1]
    #                 if y_is_neg:
    #                     x_xyz[1] = -x_xyz[1]
    #                     x_f_xyz[1] = -x_f_xyz[1]
    #                 # if third_image:
    #                 #     x_xyz[1] = -x_xyz[1]
    #                 #     x_f_xyz[1] = -x_f_xyz[1]
                        
                        
    #                 l_x.append(x_xyz)
    #                 l_results.append(r)
    #             # plt.plot(*x0_xyz[0:2], "o")
    #             # if i == 1:
    #                 # plt.plot(*x_xyz[0:2])#, c[counter]+fmt[i])

    #         if r_f > l_r_end[0]: # r_f > r(phi_low)
    #             phi_high = phi_low
    #             phi_low = phi_low - 0.1*np.pi
    #             log.update({"action":1})
    #         elif r_f < l_r_end[2]:
    #             if phi_0 < phi_f:
    #                 phi_low = phi_high
    #                 phi_high = phi_high +0.1*np.pi
    #             elif phi_0 > phi_f:
    #                 phi_high = phi_low
    #                 phi_low = phi_low -0.1*np.pi
    #             log.update({"action":2})

    #         elif r_f > l_r_end[1]:
    #             phi_high = phi_mean
    #             # phi_lo
    #             log.update({"action":3})

    #         elif r_f < l_r_end[1]:
    #             phi_low = phi_mean
    #             log.update({"action":4})
    #         else:
    #             log.update({"action":5})



    #         phi_mean = (phi_low+phi_high)/2
    #         r_high = l_r_end[0]
    #         r_low = l_r_end[2]
            
    #         # print(counter, abs(r_high-r_low)>=eps_r, abs(phi_high-phi_low)>=eps_phi, r_high, r_low)

    #         log.update({"counter": counter})
    #         log.update({"flip": flip_positions})

    #         debug_log.append(log)

    #         counter+=1

    #     #plt.plot(*x_xyz[0:2])#, c[counter]+fmt[i])

    #     if verbose: print(f"{counter=}")

    #     if flip_positions:
    #         l_k = np.array([np.flip(i, 1) for i in l_k])
    #         l_x = np.array([np.flip(i, 1) for i in l_x])



    #     # if flip_positions:
    #     #     k0_xyz = l_k[-1].T[-1]
    #     # else:
    #     #     k0_xyz = l_k[-1].T[0]
        
    #     k0_xyz = l_k[-1].T[0]


    #     x_final = l_x[-1].T
    #     length = sum([np.linalg.norm(x_final[i+1]-x_final[i]) for i in range(len(x_final)-1)])
    #     length += np.linalg.norm(x_f_xyz-x_final[-1])

    #     if debug:
    #         return k0_xyz, length, l_k, l_x, l_results, debug_log
    #     else:
    #         return k0_xyz, length, l_k, l_x, l_results



    # def geodesic_pp_2(self, x_f_xyz=np.array([-10,10,0]), x0_xyz = np.array([10,0,0]), image_nr=1, \
    #                 eps_r = 1, eps_phi=0.00000000005):

    #     y_is_neg = False
    #     if x_f_xyz[1] < 0:
    #         y_is_neg = True
    #         x_f_xyz[1] = -x_f_xyz[1]
            
        
    #     mass = 1.0
    #     time_like = True
    #     verbose = False
    #     gi = IntegratorSchwarzschildSPH2D(mass=mass, time_like=time_like, verbose = verbose)
    #     fmt = ["-x", "--", "-."]
    #     c = ["k", "b", "g", "r", "y"]

    #     if image_nr == 2:
    #         secondary_image = True
    #         third_image = False#True
    #     elif image_nr == 3:
    #         secondary_image = False
    #         third_image = True
    #     else: #if image_nr == 1:
    #         secondary_image = False
    #         third_image = False#True

    #     if secondary_image: # secondary_image
    #         # print(image_nr, not third_image, secondary_image, third_image)
    #         x_f_xyz_use = np.array([x_f_xyz[0], -x_f_xyz[1], x_f_xyz[2]])
    #     else:
    #         x_f_xyz_use = np.copy(x_f_xyz)
            
    #     r_f, _, phi_f = conversions.coord_conversion_xyz_to_sph(*x_f_xyz_use)
    #     print(r_f, phi_f, phi_f+2*np.pi)
    #     if third_image or secondary_image: # secondary_image
    #         phi_f = phi_f + 2*np.pi
         

    #     # if False:#first_image:
    #     phi_high = 4/4*np.pi
    #     phi_low = 0/2*np.pi
    #     # else:
    #     #     phi_low = -4/4*np.pi
    #     #     phi_high = 0/2*np.pi

    #     phi_mean = (phi_low+phi_high)/2
    #     r_high = 200
    #     r_low = 0

    #     counter=0
    #     while abs(r_high-r_low)>=eps_r and abs(phi_high-phi_low)>=eps_phi:# and counter<1:
    #         l_r_end = []
    #         l_k, l_x, l_results = [], [], []
    #         for i, phi in enumerate([phi_low, phi_mean, phi_high]):
    #             k0_xyz = np.array([np.cos(phi), np.sin(phi), 0])
    #             k0_xyz = k0_xyz/np.linalg.norm(k0_xyz)
    #             x0_sph, k0_sph = conversions.convert_xyz_to_sph(x0_xyz, k0_xyz)
    #             # print(phi, x0_xyz, k0_xyz)
    #             k, x, r = gi.calc_trajectory_sph(k0_sph, x0_sph, \
    #                         phi_end=phi_f,\
    #                         R_end = -1,\
    #                         curve_start = 0, curve_end = 1000, nr_points_curve = 10000, \
    #                         max_step = 1, first_step = None)
    #             # print(r)
    #             if len(r['y_events'][1]) > 0:
    #                 phi_end, r_end = r['y_events'][1][0][-1], r['y_events'][1][0][-3]
    #             else:
    #                 r_end, _, phi_end = x.T[-1]

    #             l_r_end.append(r_end)
    #             x_end, y_end, z_end = conversions.coord_conversion_sph_to_xyz(r_end, 1/2*np.pi, phi_end)
    #             # print(phi_f, phi_end,  abs(phi_high-phi_low), r_f, r_end, abs(r_high-r_low))
    #             # plt.plot(x_end, y_end, "o")
               
    #             x_xyz, k_xyz = conversions.convert_sph_to_xyz(x, k)
    #             if i == 1:
    #                 l_k.append(k_xyz)
    #                 if secondary_image:
    #                     x_xyz[1] = -x_xyz[1]
    #                 if y_is_neg:
    #                     x_xyz[1] = -x_xyz[1]
    #                     x_f_xyz[1] = -x_f_xyz[1]
    #                 # if third_image:
    #                 #     x_xyz[1] = -x_xyz[1]
    #                 #     x_f_xyz[1] = -x_f_xyz[1]
                        
                        
    #                 l_x.append(x_xyz)
    #                 l_results.append(r)
    #             # plt.plot(*x0_xyz[0:2], "o")
    #             # if i == 1:
    #                 # plt.plot(*x_xyz[0:2])#, c[counter]+fmt[i])

    #         if r_f > l_r_end[0]: # r_f > r(phi_low)
    #             phi_high = phi_low
    #             phi_low = phi_low - 0.1*np.pi
    #         elif r_f < l_r_end[2]:
    #             phi_low = phi_high
    #             phi_high = phi_high +0.1*np.pi
    #         elif r_f > l_r_end[1]:
    #             phi_high = phi_mean
    #             # phi_lo
    #         elif r_f < l_r_end[1]:
    #             phi_low = phi_mean

    #         phi_mean = (phi_low+phi_high)/2
    #         r_high = l_r_end[0]
    #         r_low = l_r_end[2]
            
    #         # print(counter, abs(r_high-r_low)>=eps_r, abs(phi_high-phi_low)>=eps_phi, r_high, r_low)

    #         counter+=1
    #     #plt.plot(*x_xyz[0:2])#, c[counter]+fmt[i])

    #     return l_k, l_x, l_results
