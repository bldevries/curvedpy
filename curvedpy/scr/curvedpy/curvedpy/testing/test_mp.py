# SuperFastPython.com
# example of running a function in another process
from time import sleep
from multiprocessing import Process
import multiprocessing as mp
import geodesic_integrator_mp as gi
import math
import numpy as np
from scipy.integrate import solve_ivp



def fun(t, U):
  x, v = U
  return [v, -x]


# a custom function that blocks for a moment
def task():
    # block for a moment
    sleep(1)
    # display a message
    print('This is from another process')
 
def wrap_solve_ivp(fun, start_stop, U_0, t_pts):
    return solve_ivp(fun, start_stop, U_0, t_eval=t_pts)

# entry point
if __name__ == '__main__':
    U_0 = [0, 1]
    start_stop = (0, 20*math.pi)
    t_pts = np.linspace(0, 15, 100)
    result = wrap_solve_ivp(fun, start_stop, U_0, t_pts)
    print(result.message)


    pool = mp.Pool(mp.cpu_count())

    # Step 2: `pool.apply` the `howmany_within_range()`
    results = [pool.apply(gi.calc_trajectory, args=()) for i in [1, 2, 3]]

    # Step 3: Don't forget to close
    pool.close()    

    # # create a process
    # process = Process(target=gi.calc_trajectory)#, args=('bob',))
    # # run the process
    # process.start()
    # # wait for the process to finish
    # print('Waiting for the process...')
    # process.join()




