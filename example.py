import event_warping

events = event_warping.read_h5_file(
    "20220124_201028_Panama_2022-01-24_20~12~11_NADIR.h5"
)

filtered_events = event_warping.without_most_active_pixels(events, ratio=0.01)

warped_events = event_warping.warp(
    filtered_events,
    velocity=(21.49 / 1e6, -0.74 / 1e6),  # px/µs
)

cumulative_map = event_warping.accumulate_warped_events_square(warped_events)

image = event_warping.render(
    cumulative_map,
    colormap_name="magma",  # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    gamma=lambda image: image,
)
image.save("cumulative.png")

image_pow2 = event_warping.render(
    cumulative_map,
    colormap_name="magma",
    gamma=lambda image: image ** (1 / 2),
)
image_pow2.save("cumulative_pow2.png")

image_pow4 = event_warping.render(
    cumulative_map,
    colormap_name="magma",
    gamma=lambda image: image ** (1 / 4),
)
image_pow4.save("cumulative_pow4.png")
