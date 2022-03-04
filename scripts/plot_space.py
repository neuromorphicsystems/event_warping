import configuration
import json
import matplotlib.pyplot
import numpy
import pathlib
import PIL.Image

dirname = pathlib.Path(__file__).resolve().parent

with open(
    dirname
    / "cache"
    / f"{configuration.name}_{configuration.heuristic}_{configuration.velocity_range[0]}_{configuration.velocity_range[1]}_{configuration.resolution}_{configuration.ratio}.json"
) as input:
    pixels = numpy.reshape(
        json.load(input), (configuration.resolution, configuration.resolution)
    )

colormap = matplotlib.pyplot.get_cmap("magma")

gamma = lambda image: image ** (1 / 2)

scaled_pixels = gamma((pixels - pixels.min()) / (pixels.max() - pixels.min()))

image = PIL.Image.fromarray(
    (colormap(scaled_pixels)[:, :, :3] * 255).astype(numpy.uint8)  # type: ignore
)

image.transpose(PIL.Image.FLIP_TOP_BOTTOM).save(
    dirname
    / "cache"
    / f"{configuration.name}_{configuration.heuristic}_{configuration.velocity_range[0]}_{configuration.velocity_range[1]}_{configuration.resolution}_{configuration.ratio}.png"
)
