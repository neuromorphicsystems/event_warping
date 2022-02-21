from __future__ import annotations
import dataclasses
import event_stream
import h5py
import matplotlib.colors
import matplotlib.pyplot
import pathlib
import numpy
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont
import typing


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
):
    colormap = matplotlib.pyplot.get_cmap(colormap_name)
    scaled_pixels = gamma(
        (cumulative_map.pixels - cumulative_map.pixels.min())
        / (cumulative_map.pixels.max() - cumulative_map.pixels.min())
    )
    image = PIL.Image.fromarray(
        (colormap(scaled_pixels)[:, :, :3] * 255).astype(numpy.uint8)  # type: ignore
    )
    return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
