import event_warping
import scipy.io as sio
import numpy as np

events = event_warping.read_es_file(
    "./recordings/20220124_201028_Panama_2022-01-24_20~12~11_NADIR.h5"
)

filtered_events = event_warping.without_most_active_pixels(events, ratio=0.001)

Vx = (2.0808552534865946e-05)*1e6
Vy = (-1.4256183976137506e-06)*1e6
warped_events = event_warping.warp(
    filtered_events,
    velocity=(Vx / 1e6, Vy / 1e6),  # px/µs
)

cumulative_map = event_warping.accumulate(
    sensor_size=(240, 180),
    events=filtered_events,
    velocity=(Vx / 1e6, Vy / 1e6),  # px/µs
)

for ratio in range(1, 5):
    image = event_warping.render(
        cumulative_map,
        colormap_name="magma",  # https://matplotlib.org/stable/gallery/color/colormap_reference.html
        gamma=lambda image: image ** (1 / ratio),
    )
    image.save("./recordings/Panama_r" + str(ratio))
