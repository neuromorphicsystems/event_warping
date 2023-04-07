import configuration
import event_warping
import pathlib

dirname = pathlib.Path(__file__).resolve().parent

(dirname / "cache").mkdir(exist_ok=True)

width, height, events = event_warping.read_es_file(dirname / f"{configuration.name}.es")
events = event_warping.without_most_active_pixels(events, ratio=configuration.ratio)

print(
    f"velocity={configuration.velocity}, heuristic={-event_warping.intensity_maximum(sensor_size=(width, height),events=events,velocity=(21.49 / 1e6, -0.74 / 1e6))}"
)

print(
    f"velocity={(0.0, 0.0)}, heuristic={-event_warping.intensity_maximum(sensor_size=(width, height),events=events,velocity=(0.0, 0.0))}"
)

optimized_velocity = event_warping.optimize_cma(
    sensor_size=(width, height),
    events=events,
    initial_velocity=(21.49 / 1e6, -0.74 / 1e6),
    initial_sigma=0.0001,
    heuristic_name=configuration.heuristic,
    iterations = 10
)

print(optimized_velocity)
