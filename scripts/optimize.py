import configuration
import event_warping
import numpy
import pathlib
import scipy.optimize
import time

dirname = pathlib.Path(__file__).resolve().parent
(dirname / "cache").mkdir(exist_ok=True)
events = event_warping.read_h5_file(dirname / f"{configuration.name}.h5")
# width, height, events = event_warping.read_h5_file(dirname / f"{configuration.name}.h5")

# events = event_warping.without_most_active_pixels(events, ratio=configuration.ratio)
print(f"{len(events)=}")

print(
    [21.49, -0.74],
    event_warping.intensity_maximum(
        sensor_size=(configuration.width, configuration.height),
        events=events,
        velocity=(21.49 / 1e6, -0.74 / 1e6),
    ),
)

print(
    [0.0, 0.0],
    event_warping.intensity_maximum(
        sensor_size=(configuration.width, configuration.height),
        events=events,
        velocity=(0.0, 0.0),
    ),
)


def callback(velocity: numpy.ndarray):
    print(f"velocity={velocity * 1e3}")


for method in ("L-BFGS-B", "SLSQP", "TNC", "Powell", "Nelder-Mead"):
    begin = time.monotonic()
    optimized_velocity = event_warping.optimize(
        sensor_size=(configuration.width, configuration.height),
        events=events,
        initial_velocity=(21.49 / 1e6, -0.74 / 1e6),
        heuristic_name=configuration.heuristic,
        method=method,
        callback=callback,
    )
    end = time.monotonic()
    print(f"{method=}, {optimized_velocity=}, {(end - begin)=} s")


#method='L-BFGS-B', optimized_velocity=array([21.24186793, -0.7537437 ]), (end - begin)=63.330848587999995 s
#method='TNC', optimized_velocity=array([21.24125378, -0.74969092]), (end - begin)=24.422812252 s
#method='Nelder-Mead', optimized_velocity=array([21.2455809, -0.7498437]), (end - begin)=21.767327560000012 s
