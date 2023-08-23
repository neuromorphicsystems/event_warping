from __future__ import annotations
import cmaes
import os
import copy
import dataclasses
import event_stream
import event_warping_extension
import h5py
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import pathlib
import numpy
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont
import scipy.optimize
import typing
from typing import Tuple

def read_es_file(path: typing.Union[pathlib.Path, str]) -> Tuple[int, int, numpy.ndarray]:
    with event_stream.Decoder(path) as decoder:
        return (
            decoder.width,
            decoder.height,
            numpy.concatenate([packet for packet in decoder]),
        )

def read_txt_file(path: typing.Union[pathlib.Path, str]) -> Tuple[int, int, numpy.ndarray]:
    dtype = [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
    events_list = []

    with open(path, 'r') as file:
        for line in file:
            parts = line.split()[:4]
            t = int(float(parts[0]) * 1e6)
            x, y, on = map(int, parts[1:])
            on = bool(on)
            events_list.append((t, x, y, on))

    events_array = numpy.array(events_list, dtype=dtype)
    width = events_array['x'].max() + 1
    height = events_array['y'].max() + 1
    return width, height, events_array


def read_h5_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    data = numpy.asarray(h5py.File(path, "r")["/FalconNeuro"], dtype=numpy.uint32)
    events = numpy.zeros(data.shape[1], dtype=event_stream.dvs_dtype)
    events["t"] = data[3]
    events["x"] = data[0]
    events["y"] = data[1]
    events["on"] = data[2] == 1
    return numpy.max(events["x"].max()) + 1, numpy.max(events["y"]) + 1, events  # type: ignore


def read_es_or_h5_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    if pathlib.Path(path).with_suffix(".es").is_file():
        return read_es_file(path=pathlib.Path(path).with_suffix(".es"))
    elif pathlib.Path(path).with_suffix(".h5").is_file():
        return read_h5_file(path=pathlib.Path(path).with_suffix(".h5"))
    raise Exception(
        f"neither \"{pathlib.Path(path).with_suffix('.es')}\" nor \"{pathlib.Path(path).with_suffix('.h5')}\" exist"
    )


@dataclasses.dataclass
class CumulativeMap:
    pixels: numpy.ndarray

def without_most_active_pixels(events: numpy.ndarray, ratio: float):
    assert ratio >= 0.0 and ratio <= 1.0
    count = numpy.zeros((events["x"].max() + 1, events["y"].max() + 1), dtype="<u8")
    numpy.add.at(count, (events["x"], events["y"]), 1)  # type: ignore
    return events[count[events["x"], events["y"]]<= numpy.percentile(count, 100.0 * (1.0 - ratio))]

def with_most_active_pixels(events: numpy.ndarray):
    return events[events["x"], events["y"]]

# velocity in px/us
def warp(events: numpy.ndarray, velocity: tuple[float, float]):
    warped_events = numpy.array(
        events, dtype=[("t", "<u8"), ("x", "<f8"), ("y", "<f8"), ("on", "?")]
    )
    warped_events["x"] -= velocity[0] * warped_events["t"]
    warped_events["y"] -= velocity[1] * warped_events["t"]
    return warped_events

def unwarp(warped_events: numpy.ndarray, velocity: tuple[float, float]):
    events = numpy.zeros(
        len(warped_events),
        dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")],
    )
    events["t"] = warped_events["t"]
    events["x"] = numpy.round(
        warped_events["x"] + velocity[0] * warped_events["t"]
    ).astype("<u2")
    events["y"] = numpy.round(
        warped_events["y"] + velocity[1] * warped_events["t"]
    ).astype("<u2")
    events["on"] = warped_events["on"]
    return events



def smooth_histogram(warped_events: numpy.ndarray):
    return event_warping_extension.smooth_histogram(warped_events)


def accumulate(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return CumulativeMap(
        pixels=event_warping_extension.accumulate(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
        ),
    )

def render(
    cumulative_map: CumulativeMap,
    colormap_name: str,
    gamma: typing.Callable[[numpy.ndarray], numpy.ndarray],
    bounds: typing.Optional[tuple[float, float]] = None,
):
    colormap = matplotlib.pyplot.get_cmap(colormap_name) # type: ignore
    if bounds is None:
        bounds = (cumulative_map.pixels.min(), cumulative_map.pixels.max())
    scaled_pixels = gamma(
        numpy.clip(
            (cumulative_map.pixels - bounds[0]) / (bounds[1] - bounds[0]),# type: ignore
            0.0,
            1.0,
        )
    )
    image = PIL.Image.fromarray(
        (colormap(scaled_pixels)[:, :, :3] * 255).astype(numpy.uint8)  # type: ignore
    )
    return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)


def render_histogram(cumulative_map: CumulativeMap, path: pathlib.Path, title: str):
    matplotlib.pyplot.figure(figsize=(16, 9))
    matplotlib.pyplot.hist(cumulative_map.pixels.flat, bins=200, log=True)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel("Event count")
    matplotlib.pyplot.ylabel("Pixel count")
    matplotlib.pyplot.savefig(path)
    matplotlib.pyplot.close()


def intensity_variance(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return event_warping_extension.intensity_variance(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )

def variance_loss_calculator(evmap: numpy.ndarray):
    flating = evmap.flatten()
    res = flating[flating != 0]
    return numpy.var(res)

def correction(i: numpy.ndarray, j: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int):
    return {
        '1': (1, vx / width, vy / height),
        '2': vx / x[i, j],
        '3': vy / y[i, j],
        '4': vx / (-x[i, j] + width + vx),
        '5': vy / (-y[i, j] + height + vy),
        '6': (vx*vy) / (vx*y[i, j] + vy*width - vy*x[i, j]),
        '7': (vx*vy) / (vx*height - vx*y[i, j] + vy*x[i, j]),
    }


def alpha_1(warped_image: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int, edgepx: int):
    """
    Inumpyut:
    warped_image: A 2D numpy array representing the warped image, where pixel values represent the event count.

    This function apply a correction on the warped image based on a set of conditions where vx < w and vy < h. The conditions are designed based on the pixel's 
    x and y positions and additional parameters (vx, vy, width, height).

    Output:
    A 2D numpy array (image) that has been corrected based on a set of conditions. This image is then fed to the Contrast Maximization algorithm
    to estimate the camera.
    """
    conditions = [
        (x > vx) & (x < width) & (y >= vy) & (y <= height),
        (x > 0) & (x < vx) & (y <= height) & (y >= ((vy*x) / vx)),
        (x >= 0) & (x <= width) & (y > 0) & (y < vy) & (y < ((vy*x) / vx)),
        (x >= width) & (x <= width+vx) & (y >= vy) & (y <= (((vy*(x-width)) / vx) + height)),
        (x > vx) & (x < width+vx) & (y > height) & (y > (((vy*(x-width)) / vx) + height)) & (y < height+vy),
        (x > width) & (x < width+vx) & (y >= ((vy*(x-width)) / vx)) & (y < vy),
        (x > 0) & (x < vx) & (y > height) & (y <= (((vy*x) / vx) + height))
    ]

    for idx, condition in enumerate(conditions, start=1):
        i, j = numpy.where(condition)            
        correction_func = correction(i, j, x, y, vx, vy, width, height)
        if idx == 1:
            warped_image[i+1, j+1] *= correction_func[str(idx)][0]
        else:    
            warped_image[i+1, j+1] *= correction_func[str(idx)]

    warped_image[x > width+vx-edgepx] = 0
    warped_image[x < edgepx] = 0
    warped_image[y > height+vy-edgepx] = 0
    warped_image[y < edgepx] = 0
    warped_image[y < ((vy*(x-width)) / vx) + edgepx] = 0
    warped_image[y > (((vy*x) / vx) + height) - edgepx] = 0
    warped_image[numpy.isnan(warped_image)] = 0
    return warped_image


def alpha_2(warped_image: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int, edgepx: int, section: int):
    """
    Inumpyut:
    warped_image: A 2D numpy array representing the warped image, where pixel values represent the event count.

    This function apply a correction on the warped image based on a set of conditions where vx > w and vy > h. The conditions are designed based on the pixel's 
    x and y positions and additional parameters (vx, vy, width, height).

    Output:
    A 2D numpy array (image) that has been corrected based on a set of conditions. This image is then fed to the Contrast Maximization algorithm
    to estimate the camera.
    """
    conditions_1 = [
        (x >= width) & (x <= vx) & (y >= (vy*x)/vx) & (y <= (vy/vx)*(x-width-vx)+vy+height), 
        (x > 0) & (x < width) & (y >= (vy*x)/vx) & (y < height), 
        (x > 0) & (x <= width) & (y > 0) & (y < (vy*x)/vx), 
        (x > vx) & (x < vx+width) & (y > vy) & (y <= (vy/vx)*(x-width-vx)+vy+height), 
        (x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width-vx)+vy+height) & (y < height+vy), 
        (x > width) & (x <= vx+width) & (y >= (vy*(x-width))/vx) & (y < vy) & (y < (vy*x)/vx) & (y > 0), 
        (x > 0) & (x <= vx) & (y < (vy/vx)*x+height) & (y >= height) & (y > (vy/vx)*(x-width-vx)+vy+height) 
    ]

    conditions_2 = [
        (x >= 0) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy) & (y > height) & (y < (vy*x)/vx), 
        (x >= 0) & (x <= vx) & (y > (vy*x)/vx) & (y < height),
        (x >= 0) & (x < width) & (y >= 0) & (y < (vy*x)/vx) & (y < height), 
        (x > width) & (x < vx+width) & (y <= ((vy*(x-width))/vx)+height) & (y > vy), 
        (x >= vx) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy+height) & (y > vy), 
        (x >= width) & (x <= vx+width) & (y > (vy/vx)*(x-width)) & (y < ((vy*(x-width))/vx)+height) & (y > 0) & (y <vy), 
        (x >= 0) & (x <= vx) & (y <= (vy/vx)*x+height) & (y > (vy/vx)*x) & (y > height) & (y <= height+vy) 
    ]

    conditions = [conditions_1, conditions_2]
    for idx, condition in enumerate(conditions[section-1], start=1):
        i, j = numpy.where(condition)
        correction_func = correction(i, j, x, y, vx, vy, width, height)
        if idx == 1:
            warped_image[i+1, j+1] *= correction_func[str(idx)][section]
        else:    
            warped_image[i+1, j+1] *= correction_func[str(idx)]

    warped_image[x > width+vx-edgepx] = 0
    warped_image[x < edgepx] = 0
    warped_image[y > height+vy-edgepx] = 0
    warped_image[y < edgepx] = 0
    warped_image[y < ((vy*(x-width)) / vx) + edgepx] = 0
    warped_image[y > (((vy*x) / vx) + height) - edgepx] = 0
    warped_image[numpy.isnan(warped_image)] = 0
    return warped_image

def mirror(warped_image: numpy.ndarray):
    mirrored_image = []
    height, width = len(warped_image), len(warped_image[0])
    for i in range(height):
        mirrored_row = []
        for j in range(width - 1, -1, -1):
            mirrored_row.append(warped_image[i][j])
        mirrored_image.append(mirrored_row)
    return numpy.array(mirrored_image)

def intensity_weighted_variance(sensor_size: tuple[int, int],events: numpy.ndarray,velocity: tuple[float, float]):
    numpy.seterr(divide='ignore', invalid='ignore')
    t               = (events["t"][-1]-events["t"][0])/1e6
    edgepx          = t
    width           = sensor_size[0]
    height          = sensor_size[1]
    fieldx          = velocity[0] / 1e-6
    fieldy          = velocity[1] / 1e-6
    velocity        = (fieldx * 1e-6, fieldy * 1e-6)
    warped_image    = accumulate(sensor_size, events, velocity)
    vx              = numpy.abs(fieldx*t)
    vy              = numpy.abs(fieldy*t)
    x               = numpy.tile(numpy.arange(1, warped_image.pixels.shape[1]+1), (warped_image.pixels.shape[0], 1))
    y               = numpy.tile(numpy.arange(1, warped_image.pixels.shape[0]+1), (warped_image.pixels.shape[1], 1)).T
    corrected_iwe   = None
    var             = 0.0
    
    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height):
        corrected_iwe            = alpha_1(warped_image.pixels, x, y, vx, vy, width, height, edgepx)
        
    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_1(warped_image.pixels, x, y, vx, vy, width, height, edgepx)
        
    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        corrected_iwe            = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 1)

    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        corrected_iwe            = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 2)

    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 1)

    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 2)
    
    if corrected_iwe is not None:
        var = variance_loss_calculator(corrected_iwe)
    return var

def intensity_maximum(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return event_warping_extension.intensity_maximum(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )

def optimize_local(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    initial_velocity: tuple[float, float],  # px/µs
    heuristic_name: str,  # max or variance
    method: str,  # Nelder-Mead, Powell, L-BFGS-B, TNC, SLSQP
    # see Constrained Minimization in https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    callback: typing.Callable[[numpy.ndarray], None],
):
    def heuristic(velocity):
        if heuristic_name == "max":
            return -intensity_maximum(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3),
            )
        elif heuristic_name == "variance":
            return -intensity_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3),
            )
        elif heuristic_name == "weighted_variance":
            return -intensity_weighted_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3))
        else:
            raise Exception(f'unknown heuristic name "{heuristic_name}"')

    if method == "Nelder-Mead":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] * 1e3, initial_velocity[1] * 1e3],
            method=method,
            bounds=scipy.optimize.Bounds([-1.0, -1.0], [1.0, 1.0]),
            options={'maxiter': 5000, "maxfev": 5000},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    elif method == "BFGS":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] / 1e2, initial_velocity[1] / 1e2],
            method=method,
            options={'ftol': 1e-9,'maxiter': 50},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    else:
        raise Exception(f'unknown optimisation method: "{method}"')

def optimize_cma(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    initial_velocity: tuple[float, float],
    initial_sigma: float,
    heuristic_name: str,
    iterations: int,
):
    def heuristic(velocity):
        if heuristic_name == "max":
            return -intensity_maximum(
                sensor_size,
                events,
                velocity=velocity,
            )
        elif heuristic_name == "variance":
            return -intensity_variance(
                sensor_size,
                events,
                velocity=velocity,
            )
        elif heuristic_name == "weighted_variance":
            return -intensity_weighted_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3))
        else:
            raise Exception(f'unknown heuristic name "{heuristic_name}"')

    optimizer = cmaes.CMA(
        mean=numpy.array(initial_velocity) * 1e3,
        sigma=initial_sigma * 1e3,
        bounds=numpy.array([[-1.0, 1.0], [-1.0, 1.0]]),
    )
    best_velocity: tuple[float, float] = copy.copy(initial_velocity)
    best_heuristic = numpy.Infinity
    for _ in range(0, iterations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = heuristic((x[0] / 1e3, x[1] / 1e3))
            solutions.append((x, value))
        optimizer.tell(solutions)
        velocity_array, heuristic_value = sorted(
            solutions, key=lambda solution: solution[1]
        )[0]
        velocity = (velocity_array[0] / 1e3, velocity_array[1] / 1e3)
        if heuristic_value < best_heuristic:
            best_velocity = velocity
            best_heuristic = heuristic_value
    return (float(best_velocity[0]), float(best_velocity[1]))
