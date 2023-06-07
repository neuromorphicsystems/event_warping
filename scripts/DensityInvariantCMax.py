import concurrent.futures
import event_warping
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from typing import Tuple


class DensityInvariantCMax:

    OBJECTIVE = {
        "variance": event_warping.intensity_variance,
        "weighted_variance": event_warping.intensity_weighted_variance,
        "max": event_warping.intensity_maximum
    }

    def __init__(self, filename: str, heuristic: str, velocity_range: Tuple[float, float], resolution: int, 
                 ratio: float, tstart: int, tfinish: int, read_path: str, save_path: str):

        self.filename       = filename
        self.heuristic      = heuristic
        self.velocity_range = velocity_range
        self.resolution     = resolution
        self.ratio          = ratio
        self.tstart         = tstart
        self.tfinish        = tfinish
        self.read_path      = read_path
        self.save_path      = save_path

        np.seterr(divide='ignore', invalid='ignore')  # Ignore divide and invalid errors

    def load_and_preprocess_events(self):
        width, height, events = event_warping.read_es_file(self.read_path + self.filename + ".es")
        events = event_warping.without_most_active_pixels(events, ratio=self.ratio)
        self.events = events[np.where(np.logical_and(events["t"] > self.tstart, events["t"] < self.tfinish))]

        print(f"Number of events: {len(self.events)}")

        self.size = (width, height)

    def calculate_heuristic(self, velocity: Tuple[float, float]):
        """Calculate the heuristic based on the method specified"""
        return self.OBJECTIVE[self.heuristic](self.size, self.events, velocity)

    def process_velocity_grid(self):
        """Compute the heuristic for a grid of velocities using concurrent processing"""
        scalar_velocities = np.linspace(*self.velocity_range, self.resolution)
        values = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for value in executor.map(
                self.calculate_heuristic,
                ((vx, vy) for vx, vy in itertools.product(scalar_velocities, scalar_velocities))
            ):
                values.append(value)
                print(f"\rProcessed {len(values)} / {self.resolution ** 2} "
                      f"({(len(values) / (self.resolution ** 2) * 100):.2f} %)", end='')
                
        return values

    def save_values(self, values: list):
        """Save the computed values to a JSON file"""
        output_file = f"{self.save_path + self.filename}_{self.heuristic}_{self.velocity_range[0]}_" \
                      f"{self.velocity_range[1]}_{self.resolution}_{self.ratio}.json"
        with open(output_file, "w") as output:
            json.dump(values, output)

        return output_file

    def generate_image(self, output_file: str):
        """Generate an image from the computed values and save it as a PNG"""
        with open(output_file) as input:
            pixels = np.reshape(json.load(input), (self.resolution, self.resolution))

        colormap = plt.get_cmap("magma")
        scaled_pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min())) ** (1 / 2)
        image = PIL.Image.fromarray((colormap(scaled_pixels)[:, :, :3] * 255).astype(np.uint8))
        resized_image = image.rotate(90, PIL.Image.NEAREST, expand=1).resize((400, 400))
        
        resized_image.save(self.save_path + self.filename + 
                           f"_{self.velocity_range[0]}_{self.velocity_range[1]}_{self.resolution}_{self.ratio}.png")

    def process(self):
        """Main method to run the processing pipeline"""
        self.load_and_preprocess_events()
        values = self.process_velocity_grid()
        output_file = self.save_values(values)
        self.generate_image(output_file)

if __name__ == "__main__":
    # Define the list of events and objectives
    EVENTS = [
        "20230112_13_aus_melbourne_nadir_day_2023-01-13_05~38~28_NADIR",
        "20220125_New_Zealand_220728_2022-01-25_22-09-42_NADIR",
        "FN034HRTEgyptB_NADIR",
        "20220124_201028_Panama_2022-01-24_20_12_11_NADIR.h5",
        "20220217_Houston_IAH_1_2022-02-17_20-28-02_NADIR"
    ]

    OBJECTIVE = ["variance","weighted_variance","max"]
    calculator = DensityInvariantCMax(filename=EVENTS[-1], heuristic=OBJECTIVE[1], velocity_range=(-30, 30), 
                                     resolution=30, ratio=0.0000001, tstart=0, tfinish = 50e6,
                                     read_path="/home/samiarja/Desktop/PhD/Code/orbital_localisation/data/es/NADIR/",
                                     save_path="/home/samiarja/Desktop/PhD/Code/orbital_localisation/img/")
    calculator.process()