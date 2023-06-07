import concurrent.futures
import event_warping
import itertools
import json
import numpy as np
import pathlib
import sys
import matplotlib.pyplot as plt
import PIL.Image
import scipy.optimize

'''
python3 -m pip install -e . && python scripts/space.py
'''

VIDEOS          = ["20220124_201028_Panama_2022-01-24_20_12_11_NADIR.h5"]
OBJECTIVE       = ["variance","weighted_variance","max"]
method          = ["SLSQP", "BFGS", "Nelder-Mead"]
SOLVER          = method[1]
HEURISTIC       = OBJECTIVE[1]
FILENAME        = VIDEOS[0]
VELOCITY_RANGE  = (-30, 30)
RESOLUTION      = 20
TMAX            = 5e6
RATIO           = 0.0000001
READFROM        = "/media/sam/Samsung_T52/PhD/Code/orbital_localisation/data/es/NADIR/"
SAVEFILETO      = "/media/sam/Samsung_T52/PhD/Code/orbital_localisation/img/"

np.seterr(divide='ignore', invalid='ignore')
width, height, events = event_warping.read_es_file(READFROM + FILENAME + ".es")
events = event_warping.without_most_active_pixels(events, ratio=0.000001)
ii = np.where(np.logical_and(events["t"]>1, events["t"]<(TMAX)))
events = events[ii]
print(f"{len(events)=}")

def calculate_heuristic(velocity: tuple[float, float]):
    if HEURISTIC == "variance":
        return event_warping.intensity_variance((width, height), events, velocity)
    if HEURISTIC == "weighted_variance":
        return event_warping.intensity_weighted_variance((width, height), events, velocity)
    if HEURISTIC == "max":
        return event_warping.intensity_maximum((width, height), events, velocity)
    raise Exception("unknown heuristic")

def callback(velocity: np.ndarray):
    print(f"velocity={velocity * 1e3}")

initial_v = (20 / 1e6, 1 / 1e6) #(21.49 / 1e6, -0.74 / 1e6)
STEP = events["t"][-1] #500000
ii = np.where(np.logical_and(events["t"]>1, events["t"]<(STEP)))

optimized_velocity = event_warping.optimize_local(sensor_size=(width, height),
                                                events=events[ii],
                                                initial_velocity=initial_v,
                                                heuristic_name=OBJECTIVE[1],
                                                method=SOLVER,
                                                callback=callback)

speed = [optimized_velocity[0],optimized_velocity[1]]
print(optimized_velocity)