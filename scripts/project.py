import configuration
import cycler
import event_warping
import matplotlib
import matplotlib.pyplot
import pathlib
import svg

matplotlib.pyplot.style.use("dark_background")
matplotlib.pyplot.rc("font", size=20, family="Times New Roman")
matplotlib.rcParams["axes.prop_cycle"] = cycler.cycler(color=["#268bd2"])
matplotlib.rcParams["axes.facecolor"] = "#1A1A1A"
matplotlib.rcParams["savefig.facecolor"] = "#1A1A1A"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.linewidth"] = 1.0

dirname = pathlib.Path(__file__).resolve().parent

(dirname / "cache").mkdir(exist_ok=True)

events = event_warping.read_es_file(dirname / f"{configuration.name}.es")
events = event_warping.without_most_active_pixels(events, ratio=configuration.ratio)
warped_events = event_warping.warp(
    events, velocity=(configuration.velocity[0] / 1e6, configuration.velocity[1] / 1e6)
)

cumulative_map = event_warping.accumulate_warped_events_square(warped_events)

event_warping.render(
    cumulative_map,
    colormap_name="magma",
    gamma=lambda image: image ** (1 / 2),
).save(
    str(
        dirname
        / "cache"
        / f"{configuration.name}_project_{configuration.velocity[0]}_{configuration.velocity[1]}_{configuration.ratio}.png"
    )
)

svg_path = (
    dirname
    / "cache"
    / f"{configuration.name}_histogram_{configuration.velocity[0]}_{configuration.velocity[1]}_{configuration.ratio}.svg"
)
event_warping.render_histogram(
    cumulative_map,
    svg_path,
    f"{configuration.name}, vx={configuration.velocity[0]}, vy={configuration.velocity[1]}, drop={configuration.ratio}",
)
svg.fix(svg_path)
