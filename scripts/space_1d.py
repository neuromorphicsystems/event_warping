import concurrent.futures
import configuration
import event_warping
import json
import numpy
import pathlib
import sys

dirname = pathlib.Path(__file__).resolve().parent


(dirname / "cache").mkdir(exist_ok=True)

events = event_warping.read_es_file(dirname / f"{configuration.name}.es")
events = event_warping.without_most_active_pixels(events, ratio=configuration.ratio)


def calculate_heuristic(velocity: float):
    warped_events = event_warping.warp(events, (velocity, velocity))
    smooth_histogram_x = event_warping.smooth_histogram(warped_events["x"])
    smooth_histogram_y = event_warping.smooth_histogram(warped_events["y"])
    if configuration.heuristic == "variance":
        return numpy.var(smooth_histogram_x), numpy.var(smooth_histogram_y)
    if configuration.heuristic == "max":
        return numpy.max(smooth_histogram_x), numpy.max(smooth_histogram_y)
    raise Exception("unknown heuristic")


with concurrent.futures.ThreadPoolExecutor() as executor:
    scalar_velocities = list(
        numpy.linspace(
            configuration.velocity_range[0],
            configuration.velocity_range[1],
            configuration.resolution,
        )
    )
    values = []
    for value in executor.map(
        calculate_heuristic, (velocity / 1e6 for velocity in scalar_velocities)
    ):
        values.append(value)
        sys.stdout.write(
            f"\r{len(values)} / {configuration.resolution} ({(len(values) / configuration.resolution * 100):.2f} %)"
        )
        sys.stdout.flush()
    sys.stdout.write("\n")
    with open(
        dirname
        / "cache"
        / f"1d_{configuration.name}_{configuration.heuristic}_{configuration.velocity_range[0]}_{configuration.velocity_range[1]}_{configuration.resolution}_{configuration.ratio}.json",
        "w",
    ) as output:
        json.dump(values, output)
