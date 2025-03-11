from kgeo.kerr_raytracing_ana import raytrace_ana
import numpy as np

# alpha = np.linspace(-100,100,200)
# beta = np.linspace(-100,100,200)#0*alpha# 

coords = np.linspace(-50,50,512)
a, b = np.meshgrid(coords, coords)
a = a.flatten()
b = b.flatten()

raytrace_ana(a=0.99, observer_coords=[0, 100000., 1*np.pi/180.,0], image_coords = [a, b], savedata=True, plotdata=False)