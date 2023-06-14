import event_warping
import scipy.io as sio
import numpy as np

iss_speed_data = [
    ["20220121a_Salvador_2022-01-21_20~58~34_NADIR.h5", 21.645, -0.5330],
    ["20220217_Houston_IAH_1_2022-02-17_20-28-02_NADIR", 21.0034, -2.7003],
    ["20220201_DIA_201410_2022-02-01_20-15-58_NADIR", 21.0000, -0.9500],
    ["20220127_Biscay_Spain_Med_211912_2_2022-01-27_21-53-58_NADIR", 20.6000, -2.2000],
    ["20220217_Sumatra_Night_2_2022-02-17_21", 20.0600, -0.7100],
    ["20230119_4_UK_Nadir_Night_2023-01-19_20~25~10_NADIR", 20.3900, -0.8050],
    ["FN034HRTEgyptB_NADIR", 20.7900, -2.5700],
    ["20220124_201028_Panama_2022-01-24_20_12_11_NADIR.h5", 21.4900, -0.7400]
]

idx = 1
TMAX = 180e6
ratio = 3
filename = iss_speed_data[idx][0]
Vx = iss_speed_data[idx][1]
Vy = iss_speed_data[idx][2]
width, height, events = event_warping.read_es_file(
    "/home/samiarja/Desktop/PhD/Code/orbital_localisation/data/es/NADIR/"
    + filename
    + ".es"
)

events = event_warping.without_most_active_pixels(events, ratio=0.000001)
ii = np.where(np.logical_and(events["t"] > 1, events["t"] < (TMAX)))
events = events[ii]

warped_events = event_warping.warp(
    events,
    velocity=(Vx / 1e6, Vy / 1e6),  # px/µs
)

cumulative_map = event_warping.accumulate(
    sensor_size=(240, 180),
    events=events,
    velocity=(Vx / 1e6, Vy / 1e6),  # px/µs
)

image = event_warping.render(
    cumulative_map,
    colormap_name="magma",  # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    gamma=lambda image: image ** (1 / ratio),
)
image.save(
    "/home/samiarja/Desktop/PhD/Code/orbital_localisation/event_warping/img/"
    + filename
    + "_landscape_"
    + str(Vx)
    + "_"
    + str(Vy)
    + ".png"
)
