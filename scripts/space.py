import concurrent.futures
import event_warping
import itertools
import json
import numpy
import pathlib
import sys
import matplotlib.pyplot
import PIL.Image


# width, height, events = event_warping.read_es_file(dirname / f"{configuration.name}.es")
VIDEOS          = ["event_100000_signal_0","20220124_201028_Panama_2022-01-24_20_12_11_NADIR.h5", "FN034HRTEgyptB_NADIR"]
OBJECTIVE       = ["variance","weighted_variance","max"]
FILENAME        = VIDEOS[0]
HEURISTIC       = OBJECTIVE[1]
VELOCITY_RANGE  = (-50.0, 50.0)
RESOLUTION      = 200
TMAX            = 30e6
RATIO           = 0.0000001
READFROM        = "/media/sam/Samsung_T51/PhD/Code/orbital_localisation/test_files/"
SAVEFILETO      = "/media/sam/Samsung_T51/PhD/Code/orbital_localisation/img/"

numpy.seterr(divide='ignore', invalid='ignore')
width, height, events = event_warping.read_es_file(READFROM + FILENAME + ".es")
events = event_warping.without_most_active_pixels(events, ratio=0.000001)
ii = numpy.where(numpy.logical_and(events["t"]>1, events["t"]<(TMAX)))
events = events[ii]
print(f"{len(events)=}")

def calculate_heuristic(velocity: tuple[float, float]):
    if HEURISTIC == "variance":
        return event_warping.intensity_variance((width, height), events, velocity)
    if HEURISTIC == "weighted_variance":
        return event_warping.intensity_weighted_variance((width, height), events, velocity)
    if HEURISTIC == "max":
        return event_warping.intensity_maximum((width, height), events, velocity)
    raise Exception("unknown heuristic")

with concurrent.futures.ThreadPoolExecutor() as executor:
    scalar_velocities = list(
        numpy.linspace(
            VELOCITY_RANGE[0],
            VELOCITY_RANGE[1],
            RESOLUTION,
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
            f"\r{len(values)} / {RESOLUTION ** 2} ({(len(values) / (RESOLUTION ** 2) * 100):.2f} %)"
        )
        sys.stdout.flush()
    sys.stdout.write("\n")
    with open(
        f"{SAVEFILETO + FILENAME}_{HEURISTIC}_{VELOCITY_RANGE[0]}_{VELOCITY_RANGE[1]}_{RESOLUTION}_{RATIO}.json",
        "w",
    ) as output:
        json.dump(values, output)

    #### MAKE FIGURE
    with open(
    f"{SAVEFILETO + FILENAME}_{HEURISTIC}_{VELOCITY_RANGE[0]}_{VELOCITY_RANGE[1]}_{RESOLUTION}_{RATIO}.json"
    ) as input:
        pixels = numpy.reshape(
            json.load(input), (RESOLUTION, RESOLUTION)
        )

    colormap = matplotlib.pyplot.get_cmap("magma")

    gamma = lambda image: image ** (1 / 2)

    scaled_pixels = gamma((pixels - pixels.min()) / (pixels.max() - pixels.min()))

    image = PIL.Image.fromarray(
        (colormap(scaled_pixels)[:, :, :3] * 255).astype(numpy.uint8)  # type: ignore
    )
    # image.show()
    image.transpose(PIL.Image.FLIP_TOP_BOTTOM).save(

        f"{SAVEFILETO + FILENAME}_{HEURISTIC}_{VELOCITY_RANGE[0]}_{VELOCITY_RANGE[1]}_{RESOLUTION}_{RATIO}.png"
    )



