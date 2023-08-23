import os
import random
import numpy as np
from typing import Tuple
import event_warping

class OptimizeCMax:
    OBJECTIVE = {
        "variance": event_warping.intensity_variance,
        "weighted_variance": event_warping.intensity_weighted_variance,
        "max": event_warping.intensity_maximum,
    }

    def __init__(
        self, 
        filename: str, 
        heuristic: str, 
        solver: str, 
        initial_speed: float,
        tstart: float,
        tfinish: float, 
        ratio: float, 
        read_path: str,
        save_path: str
    ):
        self.filename = filename
        self.heuristic = heuristic
        self.solver = solver
        self.tstart = tstart
        self.tfinish = tfinish
        self.ratio = ratio
        self.read_path = read_path
        self.save_path = save_path
        self.load_and_preprocess_events()
        initial_velocity = self.random_velocity(initial_speed)
        self.optimize(initial_velocity)

    @staticmethod
    def random_velocity(opt_range):
        return (random.uniform(-opt_range / 1e6, opt_range / 1e6), 
                random.uniform(-opt_range / 1e6, opt_range / 1e6))

    def load_and_preprocess_events(self):
        possible_extensions = [".es", ".txt"]
        
        file_extension = None
        for ext in possible_extensions:
            if os.path.exists(os.path.join(self.read_path, self.filename + ext)):
                file_extension = ext
                break
        
        if file_extension is None:
            raise ValueError(f"File with name {self.filename} not found in {self.read_path}")

        if file_extension == ".es":
            self.width, self.height, events = event_warping.read_es_file(
                os.path.join(self.read_path, self.filename + file_extension)
            )
        elif file_extension == ".txt":
            self.width, self.height, events = event_warping.read_txt_file(
                os.path.join(self.read_path, self.filename + file_extension)
            )
        else:
            raise ValueError(f"Unsupported data type: {file_extension}")

        events = event_warping.without_most_active_pixels(events, ratio=self.ratio)
        if events["t"][0] != 0:
            events["t"] = events["t"] - events["t"][0]
        self.events = events[
            np.where(
                np.logical_and(events["t"] > self.tstart, events["t"] < self.tfinish)
            )
        ]
        print(f"Number of events: {len(self.events)}")
        self.size = (self.width, self.height)

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
        print(optimized_velocity[0]*1e6, optimized_velocity[1]*1e6)
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
    OBJECTIVE = ["variance", "weighted_variance", "max"]
    EVENTS = [
        "events",
        "simple_noisy_events_with_motion",
        "2021-02-03_48_49-b0-e16.753394"
    ]
    
    solver = ["Nelder-Mead", "BFGS"]
    
    event_processor = OptimizeCMax(
        filename=EVENTS[0],
        heuristic=OBJECTIVE[0],
        solver=solver[0],
        initial_speed=200,
        tstart=0.25e6,
        tfinish=0.28e6,
        ratio=0.0,
        read_path="/home/samiarja/Desktop/PhD/Dataset/EED/what_is_background/",
        save_path="img/"
    )
