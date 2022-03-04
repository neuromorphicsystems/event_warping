import event_warping
import numpy
import pathlib
import time

dirname = pathlib.Path(__file__).resolve().parent

events = event_warping.read_es_file(dirname / "honduras.es")

begin = time.monotonic()
variance = event_warping.intensity_variance(events, (21.49 / 1e6, -0.74 / 1e6))
end = time.monotonic()
print(f"{end - begin} s", variance)

begin = time.monotonic()
print(event_warping.original_intensity_variance(events, (21.49 / 1e6, -0.74 / 1e6)))  # type: ignore
end = time.monotonic()
print(f"{end - begin} s", variance)
