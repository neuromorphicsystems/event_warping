import numpy as np
import event_warping
from typing import Tuple


class OptimizeCMax:
    def __init__(self, 
                 filename: str, 
                 heuristic: str, 
                 solver: str, 
                 tstart: float,
                 tfinish: float, 
                 ratio: float, 
                 read_from: str,
                 save_path: str):
        
        self.filename = filename
        self.heuristic = heuristic
        self.solver = solver
        self.tstart = tstart
        self.tfinish = tfinish
        self.ratio = ratio
        self.read_from = read_from
        self.save_path = save_path

        self.width, self.height, self.events = event_warping.read_es_file(
            self.read_from + self.filename + ".es"
        )
        self.events = event_warping.without_most_active_pixels(
            self.events, ratio=self.ratio
        )
        valid_indices = np.where(
            np.logical_and(self.events["t"] >= self.tstart, self.events["t"] <= self.tfinish)
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

    def optimize(self, initial_velocity: Tuple[float, float]):
        optimized_velocity = event_warping.optimize_local(
            sensor_size=(self.width, self.height),
            events=self.events,
            initial_velocity=initial_velocity,
            heuristic_name=self.heuristic,
            method=self.solver,
            callback=self.callback,
        )
        print(optimized_velocity)
        cumulative_map = event_warping.accumulate(
            sensor_size=(self.width, self.height),
            events=self.events,
            velocity=(optimized_velocity[0], optimized_velocity[0]),  # px/µs
        )
        image = event_warping.render(
            cumulative_map,
            colormap_name="magma",
            gamma=lambda image: image ** (1 / 3),
        )
        image.save(
            self.save_path
            + "motion_compensated_image_"
            + self.filename
            + f'_vx_{optimized_velocity[0]*1e6:.3f}_vy_{optimized_velocity[1]*1e6:.3f}.png'
        )
        return optimized_velocity


if __name__ == "__main__":
    filename            = ["simple_noisy_events_with_motion",
                           "2021-02-03_48_49-b0-e16.753394"]
    objective           = ["variance", "weighted_variance", "max"]
    solver              = ["Nelder-Mead", "BFGS"]
    initial_velocity    = (20 / 1e6, -1 / 1e6)

    event_processor = OptimizeCMax(filename=filename[1], 
                                   heuristic=objective[1], 
                                   solver=solver[1], 
                                   tstart=14e6, 
                                   tfinish=17e6, 
                                   ratio=0.0,
                                   read_from="data/",
                                   save_path="img/")
    event_processor.optimize(initial_velocity)
