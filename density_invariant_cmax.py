import os
import concurrent.futures
import event_warping
import itertools
import json
import numpy as np
import matplotlib.pyplot
import PIL.Image
from typing import Tuple
from PIL import ImageDraw

class DensityInvariantCMax:
    OBJECTIVE = {
        "variance": event_warping.intensity_variance,
        "weighted_variance": event_warping.intensity_weighted_variance,
        "max": event_warping.intensity_maximum,
    }

    def __init__(
        self,
        filename: str,
        heuristic: str,
        velocity_range: Tuple[float, float],
        resolution: int,
        ratio: float,
        tstart: float,
        tfinish: float,
        read_path: str,
        save_path: str,
    ):
        self.filename = filename
        self.heuristic = heuristic
        self.velocity_range = velocity_range
        self.resolution = resolution
        self.ratio = ratio
        self.tstart = tstart
        self.tfinish = tfinish
        self.read_path = read_path
        self.save_path = save_path


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
        """Calculate the heuristic based on the method specified"""
        return self.OBJECTIVE[self.heuristic](self.size, self.events, velocity)

    def process_velocity_grid(self):
        """Compute the heuristic for a grid of velocities using concurrent processing"""
        scalar_velocities = np.linspace(*self.velocity_range, self.resolution)
        values = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for value in executor.map(
                self.calculate_heuristic,
                (
                    (vx * 1e-6, vy * 1e-6)
                    for vx, vy in itertools.product(
                        scalar_velocities, scalar_velocities
                    )
                ),
            ):
                values.append(value)
                print(
                    f"\rProcessed {len(values)} / {self.resolution ** 2} "
                    f"({(len(values) / (self.resolution ** 2) * 100):.2f} %)",
                    end="",
                )

        return values

    def save_values(self, values: list):
        """Save the computed values to a JSON file"""
        output_file = (
            f"{self.save_path + self.filename}_{self.heuristic}_{self.velocity_range[0]}_"
            f"{self.velocity_range[1]}_{self.resolution}_{self.ratio}.json"
        )
        with open(output_file, "w") as output:
            json.dump(values, output)

        return output_file

    def generate_landscape(self, output_file: str):
        """Generate an image from the computed values and save it as a PNG"""
        with open(output_file) as input:
            pixels = np.reshape(json.load(input), (self.resolution, self.resolution))
        colormap = matplotlib.pyplot.colormaps["magma"]  # type:ignore
        scaled_pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min())) ** (
            1 / 2
        )
        image = PIL.Image.fromarray(
            (colormap(scaled_pixels)[:, :, :3] * 255).astype(np.uint8)
        )
        img_gray_original = image.convert('L')
        img_np_original = np.array(img_gray_original)
        max_intensity_index_original = np.argmax(img_np_original)
        max_intensity_coords_original = np.unravel_index(max_intensity_index_original, img_np_original.shape)
        scalar_velocities = np.linspace(*self.velocity_range, self.resolution)
        self.vx_original = scalar_velocities[max_intensity_coords_original[1]]
        self.vy_original = scalar_velocities[max_intensity_coords_original[0]]
        resized_image = image.rotate(90, PIL.Image.NEAREST).resize((500, 500))
        scale_factor = resized_image.width / image.width
        img_gray_resized = resized_image.convert('L')
        img_np_resized = np.array(img_gray_resized)
        max_intensity_index_resized = np.argmax(img_np_resized)
        max_intensity_coords_resized = np.unravel_index(max_intensity_index_resized, img_np_resized.shape)
        draw = ImageDraw.Draw(resized_image)
        radius = 20
        draw.ellipse((max_intensity_coords_resized[1]-radius, max_intensity_coords_resized[0]-radius, 
                    max_intensity_coords_resized[1]+radius, max_intensity_coords_resized[0]+radius), outline='black')# type:ignore
        self.vx_resized = scalar_velocities[int(max_intensity_coords_resized[1] / scale_factor)]
        text = f'vx: {self.vy_original:.3f}, vy: {self.vx_original:.3f}'
        text_size = (len(text) * 6, radius)
        if max_intensity_coords_resized[1] + radius + text_size[0] < resized_image.width:
            text_pos_x = max_intensity_coords_resized[1] + radius
        else:
            text_pos_x = max_intensity_coords_resized[1] - radius - text_size[0]
        if max_intensity_coords_resized[0] + radius + text_size[1] < resized_image.height:
            text_pos_y = max_intensity_coords_resized[0] + radius
        else:
            text_pos_y = max_intensity_coords_resized[0] - radius - text_size[1]
        draw.text((text_pos_x, text_pos_y), text, fill='white')# type:ignore
        resized_image.save(
            self.save_path
            + self.filename
            + f"_{self.velocity_range[0]}_{self.velocity_range[1]}_{self.resolution}_{self.heuristic}.png")

    def generate_motion_comp_img(self):
        #flip vx and vy, because the img was rotated by 90
        cumulative_map = event_warping.accumulate(
            sensor_size=self.size,
            events=self.events,
            velocity=(self.vy_original / 1e6, self.vx_original / 1e6),  # px/µs
        )
        image = event_warping.render(
            cumulative_map,
            colormap_name="magma",
            gamma=lambda image: image ** (1 / 3),
        )
        image.save(
            self.save_path
            + self.filename
            + f'_vx_{self.vy_original:.3f}_vy_{self.vx_original:.3f}_motion_comp_img.png'
        )

    def process(self):
        """Main method to run the processing pipeline"""
        self.load_and_preprocess_events()
        values = self.process_velocity_grid()
        output_file = self.save_values(values)
        self.generate_landscape(output_file)
        self.generate_motion_comp_img()


if __name__ == "__main__":
    # List of objective functions to use
    OBJECTIVE = ["variance", "weighted_variance", "max"]
    EVENTS = [
        "FN034HRTEgyptB_NADIR",
        "20220125_Brittany_211945_2022-01-25_21-21-18_NADIR",
        "20220124_201028_Panama_2022-01-24_20_12_11_NADIR.h5",
        "20230119_4_UK_Nadir_Night_2023-01-19_20~25~10_NADIR",
        "20220217_Houston_IAH_1_2022-02-17_20-28-02_NADIR",
        "20220121a_Salvador_2022-01-21_20~58~34_NADIR.h5",
        "20220127_Biscay_Spain_Med_211912_2_2022-01-27_21-53-58_NADIR",
        "20220201_DIA_201410_2022-02-01_20-15-58_NADIR",
        "20220125_Mexican_Coast_205735_2022-01-25_20~58~52_NADIR.h5",
        "20220125_New_Zealand_220728_2022-01-25_22-09-42_NADIR"
    ]

    filename_array = np.array(EVENTS)
    for file_name in filename_array:
        data_used = file_name
        objective_function = OBJECTIVE[0]
        event_warping.print_message(f"Processing: {data_used}", color='yellow', style='bold')
        event_warping.print_message(f"Objective function: {objective_function}", color='red', style='bold')

        calculator = DensityInvariantCMax(filename=data_used,
                                        heuristic=objective_function,
                                        velocity_range=(-30, 30),
                                        resolution=200, #the higher the better, but becomes computationally expensive!
                                        ratio=0.0,
                                        tstart=0.0e6,
                                        tfinish=30.0e6,
                                        read_path="data/",
                                        save_path="img/project_page/")
        calculator.process()
