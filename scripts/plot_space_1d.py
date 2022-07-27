import configuration
import cycler
import json
import matplotlib
import matplotlib.pyplot
import numpy
import pathlib

matplotlib.pyplot.style.use("dark_background")
matplotlib.pyplot.rc("font", size=20, family="Times New Roman")
matplotlib.rcParams["axes.prop_cycle"] = cycler.cycler(color=["#268bd2", "#b58900"])
matplotlib.rcParams["axes.facecolor"] = "#1A1A1A"
matplotlib.rcParams["savefig.facecolor"] = "#1A1A1A"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.linewidth"] = 1.0

dirname = pathlib.Path(__file__).resolve().parent

(dirname.parent / "figures").mkdir(exist_ok=True)

with open(
    dirname
    / "cache"
    / f"1d_{configuration.name}_{configuration.heuristic}_{configuration.velocity_range[0]}_{configuration.velocity_range[1]}_{configuration.resolution}_{configuration.ratio}.json"
) as input:
    values = numpy.array(json.load(input))

velocities = numpy.linspace(
    configuration.velocity_range[0],
    configuration.velocity_range[1],
    configuration.resolution,
)

figure = matplotlib.pyplot.figure(figsize=(16, 9), constrained_layout=True)
subplot = figure.add_subplot(2, 1, 1)
subplot.tick_params("x", labelbottom=False)
subplot.set_ylabel("count (x axis)")
subplot.plot(
    velocities,
    values[:, 0],
    "-",
    linewidth=1.0,
)
subplot = figure.add_subplot(2, 1, 2)
subplot.set_xlabel("velocity")
subplot.set_ylabel("count (y axis)")
subplot.plot(
    velocities,
    values[:, 1],
    "-",
    linewidth=1.0,
)
svg_path = (
    dirname.parent
    / "figures"
    / f"1d_{configuration.name}_{configuration.heuristic}_{configuration.velocity_range[0]}_{configuration.velocity_range[1]}_{configuration.resolution}_{configuration.ratio}.svg"
)
matplotlib.pyplot.savefig(str(svg_path))
matplotlib.pyplot.close()
