import event_warping
import numpy as np
import pyswarms as ps
from scipy.optimize import dual_annealing,basinhopping
import PIL.Image
import matplotlib.pyplot as plt
import json

TMAX                = 100e6
METHOD              = ["L-BFGS-B","BFGS", "Nelder-Mead"]
HEURISTIC           = ["variance","weighted_variance"]
RESOLUTION          = 100
vx_gt               = 20.79
vy_gt               = -2.57
np.seterr(divide='ignore', invalid='ignore')
FILENAME = "20220125_Mexican_Coast_205735_2022-01-25_20~58~52_NADIR.h5.es" #"20220125_Mexican_Coast_205735_2022-01-25_20~58~52_NADIR.h5.es"
width, height, events = event_warping.read_es_file("/media/sam/Samsung_T51/PhD/Code/orbital_localisation/data/es/NADIR/" + FILENAME)
width+=1
height+=1
events = event_warping.without_most_active_pixels(events, ratio=0.000001)
ii = np.where(np.logical_and(events["t"]>1, events["t"]<(TMAX)))
events = events[ii]
print(f"{len(events)=}")
deltaT = (events["t"][-1]-events["t"][0])/1e6

def callback(velocity: np.ndarray):
    print(f"velocity={velocity * 1e3}")

INITIAL_VELOCITY    = (20 / 1e6, -5 / 1e6)
optimized_velocity_variance  = event_warping.optimize_local(sensor_size=(width, height),
                                                  events=events,
                                                  initial_velocity=INITIAL_VELOCITY,
                                                  heuristic_name=HEURISTIC[0],
                                                  method="Nelder-Mead",
                                                  callback=callback)

optimized_velocity_weighted_variance  = event_warping.optimize_local(sensor_size=(width, height),
                                                  events=events,
                                                  initial_velocity=INITIAL_VELOCITY,
                                                  heuristic_name=HEURISTIC[1],
                                                  method="Nelder-Mead",
                                                  callback=callback)

ErrorX = np.abs(optimized_velocity_variance[0]*1e6 - vx_gt)
ErrorY = np.abs(optimized_velocity_variance[1]*1e6 - vy_gt)
print("{} vx_est: {}   vy_est: {}    vx_er: {}%    vy_er: {}%".format(HEURISTIC[0],
                                                                round(optimized_velocity_variance[0]*1e6,4),
                                                                round(optimized_velocity_variance[1]*1e6,4),
                                                                round(ErrorX*deltaT,4),
                                                                round(ErrorY*deltaT,4)))

ErrorX = np.abs(optimized_velocity_weighted_variance[0]*1e6 - vx_gt)
ErrorY = np.abs(optimized_velocity_weighted_variance[1]*1e6 - vy_gt)
print("{} vx_est: {}   vy_est: {}    vx_er: {}%    vy_er: {}%".format(HEURISTIC[1],
                                                                round(optimized_velocity_weighted_variance[0]*1e6,4),
                                                                round(optimized_velocity_weighted_variance[1]*1e6,4),
                                                                round(ErrorX*deltaT,4),
                                                                round(ErrorY*deltaT,4)))

#### MAKE FIGURE
with open("/media/sam/Samsung_T51/PhD/Code/orbital_localisation/test_files/FN034HRTEgyptB_NADIR.es_weightVar_-50_50_100_1e-05_0.json") as input:
    pixels = np.reshape(json.load(input), (RESOLUTION, RESOLUTION))
with open("/media/sam/Samsung_T51/PhD/Code/orbital_localisation/test_files/FN034HRTEgyptB_NADIR.es_weightVar_-50_50_100_1e-05_1.json") as input:
    pixels_w = np.reshape(json.load(input), (RESOLUTION, RESOLUTION))

x = np.tile(np.arange(-pixels.shape[1], pixels.shape[1],2), (pixels.shape[0], 1))
y = np.tile(np.arange(-pixels.shape[1], pixels.shape[0],2), (pixels.shape[1], 1)).T

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(x, y, pixels+1000,edgecolor='royalblue', lw=1, rstride=8, cstride=8,alpha=0.3)
ax.contour(x, y, pixels, zdir='z', offset=0, cmap='gist_rainbow')
ax.set_title('Before Correction')
ax.view_init(elev=17, azim=36)
plt.plot(optimized_velocity_variance[0]*1e6, optimized_velocity_variance[1]*1e6, 'o', color="black", label="global minimum")
plt.legend()
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(x, y, pixels_w+1000,edgecolor='royalblue', lw=1, rstride=8, cstride=8,alpha=0.3)
ax.contour(x, y, pixels_w, zdir='z', offset=0, cmap='gist_rainbow')
ax.set_title('After Correction')
ax.view_init(elev=17, azim=36)
plt.plot(optimized_velocity_weighted_variance[0]*1e6, optimized_velocity_weighted_variance[1]*1e6, 'o', color="black", label="global minimum")
plt.legend()
plt.show()