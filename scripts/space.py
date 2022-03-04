import concurrent.futures
import configuration
import event_warping
import itertools
import json
import numpy
import pathlib
import sys

dirname = pathlib.Path(__file__).resolve().parent


(dirname / "cache").mkdir(exist_ok=True)

events = event_warping.read_es_file(dirname / f"{configuration.name}.es")
events = event_warping.without_most_active_pixels(events, ratio=configuration.ratio)


def calculate_heuristic(velocity: tuple[float, float]):
    if configuration.heuristic == "variance":
        return event_warping.intensity_variance(events, velocity)
    if configuration.heuristic == "max":
        return event_warping.intensity_maximum(events, velocity)
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
        calculate_heuristic,
        (
            (velocity[1] * 1e-6, velocity[0] * 1e-6)
            for velocity in itertools.product(scalar_velocities, scalar_velocities)
        ),
    ):
        values.append(value)
        sys.stdout.write(
            f"\r{len(values)} / {configuration.resolution ** 2} ({(len(values) / (configuration.resolution ** 2) * 100):.2f} %)"
        )
        sys.stdout.flush()
    sys.stdout.write("\n")
    with open(
        dirname
        / "cache"
        / f"{configuration.name}_{configuration.heuristic}_{configuration.velocity_range[0]}_{configuration.velocity_range[1]}_{configuration.resolution}_{configuration.ratio}.json",
        "w",
    ) as output:
        json.dump(values, output)
