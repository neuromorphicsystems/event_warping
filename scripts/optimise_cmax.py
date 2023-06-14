import numpy as np
import event_warping
from typing import Tuple


class OptimizeCMax:
    def __init__(self, 
                 filename: str, 
                 heuristic: str, 
                 solver: str, 
                 tmax: float, 
                 ratio: float, 
                 read_from: str):
        
        self.filename = filename
        self.heuristic = heuristic
        self.solver = solver
        self.tmax = tmax
        self.ratio = ratio
        self.read_from = read_from

        self.width, self.height, self.events = event_warping.read_es_file(
            self.read_from + self.filename + ".es"
        )
        self.events = event_warping.without_most_active_pixels(
            self.events, ratio=self.ratio
        )
        valid_indices = np.where(
            np.logical_and(self.events["t"] >= 0, self.events["t"] <= self.tmax)
        )
        self.events = self.events[valid_indices]

    def calculate_heuristic(self, velocity: Tuple[float, float]):
        if self.heuristic == "variance":
            return event_warping.intensity_variance(
                (self.width, self.height), self.events, velocity
            )
        if self.heuristic == "weighted_variance":
            return event_warping.intensity_weighted_variance(
                (self.width, self.height), self.events, velocity
            )
        if self.heuristic == "max":
            return event_warping.intensity_maximum(
                (self.width, self.height), self.events, velocity
            )
        raise Exception("unknown heuristic")

    def callback(self, velocity: np.ndarray):
        print(f"velocity={velocity * 1e3}")

    def optimize(self, initial_velocity):
        optimized_velocity = event_warping.optimize_local(
            sensor_size=(self.width, self.height),
            events=self.events,
            initial_velocity=initial_velocity,
            heuristic_name=self.heuristic,
            method=self.solver,
            callback=self.callback,
        )
        print(optimized_velocity)
        return optimized_velocity


if __name__ == "__main__":
    filename            = ["simple_noisy_events_with_motion"]
    objective           = ["variance", "weighted_variance", "max"]
    solver              = ["Nelder-Mead", "BFGS"]
    initial_velocity    = (20 / 1e6, -1 / 1e6)

    event_processor = OptimizeCMax(filename=filename[0], 
                                   heuristic=objective[0], 
                                   solver=solver[1], 
                                   tmax=10e6, 
                                   ratio=0.0, 
                                   read_from="data/")

    event_processor.optimize(initial_velocity)
