from __future__ import annotations
import cmaes
import copy
import dataclasses
import event_stream
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

EXTENSION_ENABLED = False


def read_h5_file(path: typing.Union[pathlib.Path, str]) -> numpy.ndarray:
    data = numpy.asarray(h5py.File(path, "r")["/FalconNeuro"], dtype=numpy.uint32)
    events = numpy.zeros(data.shape[1], dtype=event_stream.dvs_dtype)
    events["t"] = data[3]
    events["x"] = data[0]
    events["y"] = data[1]
    events["on"] = data[2] == 1
    return events


def read_es_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    with event_stream.Decoder(path) as decoder:
        return (
            decoder.width,
            decoder.height,
            numpy.concatenate([packet for packet in decoder]),
        )


@dataclasses.dataclass
class CumulativeMap:
    pixels: numpy.ndarray


def without_most_active_pixels(events: numpy.ndarray, ratio: float):
    assert ratio >= 0.0 and ratio <= 1.0
    count = numpy.zeros((events["x"].max() + 1, events["y"].max() + 1), dtype="<u8")
    numpy.add.at(count, (events["x"], events["y"]), 1)  # type: ignore
    return events[
        count[events["x"], events["y"]]
        <= numpy.percentile(count, 100.0 * (1.0 - ratio))
    ]


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
    raise NotImplementedError()


def accumulate(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    raise NotImplementedError()


def render(
    cumulative_map: CumulativeMap,
    colormap_name: str,
    gamma: typing.Callable[[numpy.ndarray], numpy.ndarray],
    bounds: typing.Optional[tuple[float, float]] = None,
):
    colormap = matplotlib.pyplot.get_cmap(colormap_name)
    if bounds is None:
        bounds = (cumulative_map.pixels.min(), cumulative_map.pixels.max())
    scaled_pixels = gamma(
        numpy.clip(
            (cumulative_map.pixels - bounds[0]) / (bounds[1] - bounds[0]),
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
    raise NotImplementedError()


def intensity_maximum(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    raise NotImplementedError()


def optimize(
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
        else:
            raise Exception(f'unnknown heuristic name "{heuristic_name}"')

    result = scipy.optimize.minimize(
        fun=heuristic,
        x0=[initial_velocity[0] * 1e3, initial_velocity[1] * 1e3],
        method=method,
        bounds=scipy.optimize.Bounds([-1.0, -1.0], [1.0, 1.0]),
        callback=callback,
    ).x
    return (float(result[0]), float(result[1]))


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
        else:
            raise Exception(f'unnknown heuristic name "{heuristic_name}"')

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
            value = heuristic((x[0] * 1e-3, x[1] * 1e-3))
            solutions.append((x, value))
        optimizer.tell(solutions)
        velocity_array, heuristic_value = sorted(
            solutions, key=lambda solution: solution[1]
        )[0]
        velocity = (velocity_array[0] * 1e-3, velocity_array[1] * 1e-3)
        display_velocity = (velocity_array[0] * 1e3, velocity_array[1] * 1e3)
        if heuristic_value < best_heuristic:
            best_velocity = velocity
            best_heuristic = heuristic_value
    return (float(best_velocity[0]), float(best_velocity[1]))


# monkey patch the extension
try:
    import event_warping_extension  # type: ignore
    import sys

    for function_name in (
        "smooth_histogram",
        "accumulate",
        "intensity_variance",
        "intensity_maximum",
    ):
        getattr(event_warping_extension, function_name)
        setattr(
            sys.modules[__name__],
            f"original_{function_name}",
            getattr(sys.modules[__name__], function_name),
        )

    def accelerated_accumulate(
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

    def accelerated_intensity_variance(
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

    def accelerated_intensity_maximum(
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

    setattr(
        sys.modules[__name__],
        "smooth_histogram",
        event_warping_extension.smooth_histogram,
    )
    setattr(
        sys.modules[__name__],
        "accumulate",
        accelerated_accumulate,
    )
    setattr(
        sys.modules[__name__],
        "intensity_variance",
        accelerated_intensity_variance,
    )
    setattr(
        sys.modules[__name__],
        "intensity_maximum",
        accelerated_intensity_maximum,
    )
    sys.modules[__name__].__dict__["EXTENSION_ENABLED"] = True
except (AttributeError, ImportError):
    pass
