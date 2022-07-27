import event_warping
import pathlib

dirname = pathlib.Path(__file__).resolve().parent

(dirname / "figures").mkdir(exist_ok=True)

width, height, events = event_warping.read_es_or_h5_file(
    dirname / "recordings" / "20220124_201028_Panama_2022-01-24_20~12~11_NADIR"
)

filtered_events = event_warping.without_most_active_pixels(events, ratio=0.01)

cumulative_map = event_warping.accumulate(
    sensor_size=(240, 180),
    events=filtered_events,
    velocity=(21.49 / 1e6, -0.74 / 1e6),  # px/µs
)

# see https://matplotlib.org/stable/gallery/color/colormap_reference.html
# for a list of available colormaps
colormap_name = "magma"

image = event_warping.render(
    cumulative_map,
    colormap_name=colormap_name,
    gamma=lambda image: image,
)
image.save(str(dirname / "figures" / "cumulative.png"))

image_pow2 = event_warping.render(
    cumulative_map,
    colormap_name=colormap_name,
    gamma=lambda image: image ** (1 / 2),
)
image_pow2.save(str(dirname / "figures" / "cumulative_pow2.png"))

image_pow4 = event_warping.render(
    cumulative_map,
    colormap_name=colormap_name,
    gamma=lambda image: image ** (1 / 4),
)
image_pow4.save(str(dirname / "figures" / "cumulative_pow4.png"))
