import sympy as sp
import numpy as np
import PIL.Image
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

sp.init_printing()
 
#Declare symbolic variables explicitly which are to be treated as symbols
x,rho,vx,w,h,t  = sp.symbols("x rho vx w h t")

f1_xy = [((rho*x)/vx),
        (rho*t),
        (rho*((-x+w+vx*t)/(vx)))
        ]

limx_start_f1  = [0, vx*t,w]
limx_finish_f1 = [vx*t,w,w+vx*t]

meanf =  (sp.integrate(f1_xy[0],  (x, limx_start_f1[0],  limx_finish_f1[0]))+
          sp.integrate(f1_xy[1],  (x, limx_start_f1[1],  limx_finish_f1[1]))+
          sp.integrate(f1_xy[2],  (x, limx_start_f1[2],  limx_finish_f1[2]))
          )/((w+vx*t))
          
for idx in range(len(f1_xy)):
    variance_f2 = sp.simplify(sp.integrate(((f1_xy[idx]-meanf)**2),  (x, limx_start_f1[idx],  limx_finish_f1[idx])))
    print(sp.simplify(variance_f2))


