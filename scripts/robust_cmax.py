import os
import random
import numpy as np
import event_warping
from tqdm import tqdm
from typing import Tuple, List

class RobustCMax:
    def __init__(self, path: str, tstart: float, tfinish: float):
        self.path = path
        self.tstart = tstart * 1e6
        self.tfinish = tfinish * 1e6

        self.read_path, filename_with_ext = os.path.split(path)
        self.filename, _ = os.path.splitext(filename_with_ext)
        
        self.load_and_preprocess_events()
        self.vx, self.vy, self.contrast = self.find_best_velocity_with_iteratively(increment=10)

    @classmethod
    def run(cls, path: str, tstart: float, tfinish: float):
        instance = cls(path, tstart, tfinish)
        return instance.warp_image()

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

        events = event_warping.without_most_active_pixels(events, ratio=0.0)
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
        return event_warping.intensity_weighted_variance((self.width, self.height), self.events, velocity)

    def warp_image(self):
        cumulative_map = event_warping.accumulate(
            sensor_size=(self.width, self.height),
            events=self.events,
            velocity=(self.vx, self.vy),  # px/µs
        )
        warped_img = event_warping.render(
            cumulative_map,
            colormap_name="magma",
            gamma=lambda image: image ** (1 / 3),
        )
        return self.vx, self.vy, warped_img
    
    def callback(self, velocity: np.ndarray):
        print(f"velocity={velocity * 1e3}")

    def find_best_velocity_with_iteratively(self, increment=1):
        """
        Finds the best optimized velocity over a number of iterations.
        """
        best_velocity = None
        highest_variance = float('-inf')
        
        variances = []
        
        for vy in tqdm(range(-40, 41, increment)):
            for vx in range(-40, 41, increment):
                current_velocity = (vx / 1e6, vy / 1e6)
                
                optimized_velocity = event_warping.optimize_local(sensor_size=self.size,
                                                    events=self.events,
                                                    initial_velocity=current_velocity,
                                                    heuristic_name="weighted_variance",
                                                    method="Nelder-Mead",
                                                    callback=self.callback)
                
                warped_accumulated_img = event_warping.accumulate(sensor_size=(self.width, self.height),
                                                          events=self.events,
                                                          velocity=optimized_velocity)
                objective_loss = np.var(warped_accumulated_img.pixels.flatten())
                variances.append((optimized_velocity, objective_loss))
                
                if objective_loss > highest_variance:
                    highest_variance = objective_loss
                    best_velocity = optimized_velocity
        
        # Converting variances to a numpy array for easier handling
        variances = np.array(variances, dtype=[('velocity', float, 2), ('variance', float)])
        print(f"vx: {best_velocity[0] * 1e6} vy: {best_velocity[1] * 1e6} contrast: {highest_variance}")
        return best_velocity[0], best_velocity[1], highest_variance

