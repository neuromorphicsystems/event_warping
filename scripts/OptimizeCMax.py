import numpy as np
import event_warping
from typing import Tuple

class OptimizeCMax:

    def __init__(self, video_file, objective, solver, velocity_range, resolution, tmax, ratio, read_from, save_to):
        self.filename = video_file
        self.heuristic = objective
        self.solver = solver
        self.velocity_range = velocity_range
        self.resolution = resolution
        self.tmax = tmax
        self.ratio = ratio
        self.read_from = read_from
        self.save_to = save_to
        self.events = None
        self.width = None
        self.height = None
        self._read_and_process_events()

    def _read_and_process_events(self):
        np.seterr(divide='ignore', invalid='ignore')
        self.width, self.height, self.events = event_warping.read_es_file(self.read_from + self.filename + ".es")
        self.events = event_warping.without_most_active_pixels(self.events, ratio=self.ratio)
        valid_indices = np.where(np.logical_and(self.events["t"] > 1, self.events["t"] < self.tmax))
        self.events = self.events[valid_indices]
        print(f"{len(self.events)=}")

    def calculate_heuristic(self, velocity: Tuple[float, float]):
        if self.heuristic == "variance":
            return event_warping.intensity_variance((self.width, self.height), self.events, velocity)
        if self.heuristic == "weighted_variance":
            return event_warping.intensity_weighted_variance((self.width, self.height), self.events, velocity)
        if self.heuristic == "max":
            return event_warping.intensity_maximum((self.width, self.height), self.events, velocity)
        raise Exception("unknown heuristic")

    def callback(self, velocity: np.ndarray):
        print(f"velocity={velocity * 1e3}")

    def optimize(self, initial_velocity):
        batch = self.events["t"][-1]
        valid_indices = np.where(np.logical_and(self.events["t"] > 1, self.events["t"] < batch))
        optimized_velocity = event_warping.optimize_local(sensor_size=(self.width, self.height),
                                                          events=self.events[valid_indices],
                                                          initial_velocity=initial_velocity,
                                                          heuristic_name=self.heuristic,
                                                          method=self.solver,
                                                          callback=self.callback)
        print(optimized_velocity)
        return optimized_velocity


if __name__ == "__main__":
    video_file          = ["20220124_201028_Panama_2022-01-24_20_12_11_NADIR.h5"]
    objective           = ["variance","weighted_variance","max"]
    solver              = ["SLSQP", "BFGS", "Nelder-Mead"]
    velocity_range      = (-30, 30)
    resolution          = 20
    tmax                = 10e6
    ratio               = 0.0000001
    read_from           = "/home/samiarja/Desktop/PhD/Code/orbital_localisation/data/es/NADIR/"
    save_to             = "/home/samiarja/Desktop/PhD/Code/orbital_localisation/img/"
    initial_velocity    = (20 / 1e6, -1 / 1e6)

    event_processor = OptimizeCMax(video_file[0], 
                                   objective[1], 
                                   solver[-1], 
                                   velocity_range, 
                                   resolution, 
                                   tmax, 
                                   ratio, 
                                   read_from, 
                                   save_to)
    
    event_processor.optimize(initial_velocity)
