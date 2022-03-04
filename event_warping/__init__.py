from __future__ import annotations
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


def read_es_file(path: typing.Union[pathlib.Path, str]) -> numpy.ndarray:
    with event_stream.Decoder(path) as decoder:
        return numpy.concatenate([packet for packet in decoder])


@dataclasses.dataclass
class CumulativeMap:
    pixels: numpy.ndarray
    offset: tuple[float, float]


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


def accumulate_warped_events_gaussian(warped_events: numpy.ndarray, sigma: float):
    padding = int(numpy.ceil(6.0 * sigma))
    kernel_indices = numpy.zeros((padding * 2 + 1, padding * 2 + 1, 2))
    for y in range(0, padding * 2 + 1):
        for x in range(0, padding * 2 + 1):
            kernel_indices[y, x][0] = x - padding
            kernel_indices[y, x][1] = y - padding
    x_minimum = float(warped_events["x"].min())
    y_minimum = float(warped_events["y"].min())
    xs = warped_events["x"].astype("<f8") - x_minimum + padding
    ys = warped_events["y"].astype("<f8") - y_minimum + padding
    pixels = numpy.zeros(
        (
            int(numpy.ceil(ys.max())) + padding + 1,
            int(numpy.ceil(xs.max())) + padding + 1,
        )
    )
    xis = numpy.round(xs).astype("<i8")
    yis = numpy.round(ys).astype("<i8")
    xfs = xs - xis
    yfs = ys - yis
    sigma_factor = -1.0 / (2.0 * sigma**2.0)
    for xi, yi, xf, yf in zip(xis, yis, xfs, yfs):
        pixels[
            yi - padding : yi + padding + 1,
            xi - padding : xi + padding + 1,
        ] += numpy.exp(
            numpy.sum((kernel_indices - numpy.array([xf, yf])) ** 2.0, axis=2)
            * sigma_factor
        )
    return CumulativeMap(
        pixels=pixels,
        offset=(-x_minimum + padding, -y_minimum + padding),
    )


def smooth_histogram(warped_events: numpy.ndarray):
    raise NotImplementedError()


# accumulate_warped_events_square is a 2D version of smooth_histogram
def accumulate_warped_events_square(warped_events: numpy.ndarray):
    x_minimum = float(warped_events["x"].min())
    y_minimum = float(warped_events["y"].min())
    xs = warped_events["x"].astype("<f8") - x_minimum + 1.0
    ys = warped_events["y"].astype("<f8") - y_minimum + 1.0
    pixels = numpy.zeros((int(numpy.ceil(ys.max())) + 2, int(numpy.ceil(xs.max())) + 2))
    xis = numpy.floor(xs).astype("<i8")
    yis = numpy.floor(ys).astype("<i8")
    xfs = xs - xis
    yfs = ys - yis
    for xi, yi, xf, yf in zip(xis, yis, xfs, yfs):
        pixels[yi, xi] += (1.0 - xf) * (1.0 - yf)
        pixels[yi, xi + 1] += xf * (1.0 - yf)
        pixels[yi + 1, xi] += (1.0 - xf) * yf
        pixels[yi + 1, xi + 1] += xf * yf
    return CumulativeMap(
        pixels=pixels,
        offset=(-x_minimum + 1.0, -y_minimum + 1.0),
    )


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


def intensity_variance(events: numpy.ndarray, velocity: tuple[float, float]):
    warped_events = warp(events, velocity)
    cumulative_map = accumulate_warped_events_square(warped_events)
    return float(numpy.var(cumulative_map.pixels))


def intensity_maximum(events: numpy.ndarray, velocity: tuple[float, float]):
    warped_events = warp(events, velocity)
    cumulative_map = accumulate_warped_events_square(warped_events)
    return float(numpy.max(cumulative_map.pixels))


# monkey patch the extension
try:
    import event_warping_extension  # type: ignore
    import sys

    for function_name in (
        "smooth_histogram",
        "accumulate_warped_events_square",
        "intensity_variance",
        "intensity_maximum",
    ):
        getattr(event_warping_extension, function_name)
        setattr(
            sys.modules[__name__],
            f"original_{function_name}",
            getattr(sys.modules[__name__], function_name),
        )

    def accelerated_accumulate_warped_events_square(warped_events: numpy.ndarray):
        return CumulativeMap(
            pixels=event_warping_extension.accumulate_warped_events_square(  # type: ignore
                warped_events["x"].astype("<f8"),
                warped_events["y"].astype("<f8"),
            ),
            offset=(-warped_events["x"].min() + 1.0, -warped_events["y"].min() + 1.0),
        )

    def accelerated_intensity_variance(
        events: numpy.ndarray, velocity: tuple[float, float]
    ):
        return event_warping_extension.intensity_variance(  # type: ignore
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
        )

    def accelerated_intensity_maximum(
        events: numpy.ndarray, velocity: tuple[float, float]
    ):
        return event_warping_extension.intensity_maximum(  # type: ignore
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
        "accumulate_warped_events_square",
        accelerated_accumulate_warped_events_square,
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
